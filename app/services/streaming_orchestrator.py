"""
Streaming version of the LLM orchestrator.

Replaces the blocking process_message() call with an async generator
that yields SSE event dicts as each pipeline stage completes.

Pipeline stages and their events:
  1. Layer 1 (Claude input processor)
       → yields: thinking(layer1), then question | proceeds to prediction
  2. Layer 2 (Decision Tree, fast, no streaming needed)
       → yields: thinking(layer2)
  3. Layer 3 (Claude output validator, streamed)
       → yields: thinking(layer3), token*, _layer3_complete
  4. Web search fallback (streamed)
       → yields: thinking(searching), token*, _fallback_complete
  5. Done
       → yields: diagnosis, done

Private event types (_layer3_complete, _fallback_complete) are internal
signals between generator stages and are never forwarded to the frontend.
"""

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.core.json_utils import extract_json_from_response
from app.core.llm_input_processor import extract_symptoms_from_text
from app.core.llm_output_validator import (
    LAYER3_SYSTEM_PROMPT,
    MEDICAL_DISCLAIMER,
    dampen_confidence,
)
from app.services.predictor_service import predict_disease

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database.models.session_log import ChatSession

logger = logging.getLogger(__name__)

# Module-level async Anthropic client — created once, reused across requests.
_aclient = AsyncAnthropic(api_key=settings.anthropic_api_key)


def _event(type_: str, data: dict) -> dict:
    """Build a typed SSE event dict.

    Args:
        type_: Event type string (thinking, question, token, diagnosis, error, done).
        data:  Event payload dict.

    Returns:
        Dict with shape ``{"type": str, "data": dict}``.
    """
    return {"type": type_, "data": data}


async def stream_message(
    session: "ChatSession",
    user_message: str,
    db: "AsyncSession",
) -> AsyncGenerator[dict, None]:
    """Main streaming entry point.  Yields SSE event dicts.

    Handles COLLECTING and CONFIRMING states.  The GREETING state is
    handled by the controller before this function is called.

    Args:
        session:      ChatSession ORM object (mutated in-place).
        user_message: Raw text from the user.
        db:           Async SQLAlchemy session for persistence.  Kept open
                      for the full lifetime of the generator by FastAPI's
                      ``yield``-dependency lifecycle.

    Yields:
        Dict with shape ``{"type": str, "data": dict}``.
    """
    from app.services import session_service

    state = session.state
    confirmed: list[str] = list(session.confirmed_symptoms or [])
    denied: list[str] = list(session.denied_symptoms or [])
    history: list[dict] = list(session.conversation_history or [])
    questions_asked: int = session.questions_asked or 0
    session_id = str(session.id)

    history.append({"role": "user", "content": user_message})

    # Immediate heartbeat — frontend knows we received the message.
    yield _event("thinking", {"stage": "layer1", "message": "Dr. Melvis is listening..."})

    # ── COLLECTING state ─────────────────────────────────────────────────────
    if state == "COLLECTING":

        try:
            layer1_result = extract_symptoms_from_text(
                user_message=user_message,
                conversation_history=history[:-1],  # exclude just-appended user message
                confirmed_so_far=confirmed,
                denied_so_far=denied,
                questions_asked=questions_asked,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error(
                "Layer 1 failed in streaming: %s | session=%s", exc, session_id
            )
            yield _event("error", {"message": "Something went wrong. Please try again."})
            yield _event("done", {"session_id": session_id})
            return

        action = layer1_result.get("action")

        # Merge all confirmed/denied fields returned by Layer 1 (both the
        # legacy confirmed_symptoms/denied_symptoms fields and the newer
        # newly_confirmed/newly_denied fields from ask actions).
        for s in (
            (layer1_result.get("confirmed_symptoms") or [])
            + (layer1_result.get("newly_confirmed") or [])
        ):
            if s and s not in confirmed:
                confirmed.append(s)
        for s in (
            (layer1_result.get("denied_symptoms") or [])
            + (layer1_result.get("newly_denied") or [])
        ):
            if s and s not in denied:
                denied.append(s)

        if action == "ask":
            question = layer1_result.get(
                "question", "Could you tell me more about how you are feeling?"
            )
            history.append({"role": "assistant", "content": question})
            session.state = "COLLECTING"
            session.confirmed_symptoms = confirmed
            session.denied_symptoms = denied
            session.conversation_history = history
            session.questions_asked = questions_asked + 1
            await session_service.save_session(db, session)
            yield _event("question", {"message": question})
            yield _event("done", {"session_id": session_id})
            return

        if action in ("ready", "extract") and len(confirmed) >= 3:
            summary = layer1_result.get(
                "summary", f"Symptoms reported: {', '.join(confirmed)}"
            )
            session.confirmed_symptoms = confirmed
            session.denied_symptoms = denied
            session.conversation_history = history
            async for event in _stream_prediction(
                session, confirmed, denied, summary, history, db, session_id
            ):
                yield event
            return

        if action == "extract" and len(confirmed) < 3:
            unmapped = layer1_result.get("unmapped_complaints") or []
            if unmapped:
                follow_up = (
                    f"I noted you mentioned {unmapped[0]}. Could you tell me a bit more — "
                    f"how long have you had this, and is it constant or does it come and go?"
                )
            else:
                follow_up = (
                    "Could you describe what you're feeling in a bit more detail? "
                    "For example, how long has this been going on and how severe is it?"
                )
            history.append({"role": "assistant", "content": follow_up})
            session.state = "COLLECTING"
            session.confirmed_symptoms = confirmed
            session.denied_symptoms = denied
            session.conversation_history = history
            session.questions_asked = questions_asked + 1
            await session_service.save_session(db, session)
            yield _event("question", {"message": follow_up})
            yield _event("done", {"session_id": session_id})
            return

        # ── UNCLEAR handler with loop-break ──────────────────────────────────
        last_assistant = next(
            (
                m["content"]
                for m in reversed(history[:-1])
                if m["role"] == "assistant"
            ),
            "",
        )
        already_asked_to_clarify = (
            "didn't quite catch" in last_assistant.lower()
            or "describe what you're feeling" in last_assistant.lower()
            or "could you describe" in last_assistant.lower()
            or "make sure i understand" in last_assistant.lower()
        )

        if already_asked_to_clarify or questions_asked >= 5 or len(confirmed) >= 3:
            logger.info(
                "UNCLEAR loop broken — proceeding to prediction | session=%s", session_id
            )
            summary = (
                f"Patient described: "
                f"{', '.join(confirmed) if confirmed else 'unspecified symptoms'}. "
                f"Some responses could not be fully interpreted."
            )
            session.confirmed_symptoms = confirmed
            session.denied_symptoms = denied
            session.conversation_history = history
            async for event in _stream_prediction(
                session, confirmed, denied, summary, history, db, session_id
            ):
                yield event
        else:
            follow_up = (
                "I want to make sure I understand you correctly. "
                "Could you tell me a little more about how you are feeling?"
            )
            history.append({"role": "assistant", "content": follow_up})
            session.state = "COLLECTING"
            session.confirmed_symptoms = confirmed
            session.denied_symptoms = denied
            session.conversation_history = history
            session.questions_asked = questions_asked + 1
            await session_service.save_session(db, session)
            yield _event("question", {"message": follow_up})
            yield _event("done", {"session_id": session_id})
        return

    # ── CONFIRMING state ─────────────────────────────────────────────────────
    if state == "CONFIRMING":
        summary = f"Confirmed symptoms: {', '.join(confirmed)}"
        async for event in _stream_prediction(
            session, confirmed, denied, summary, history, db, session_id
        ):
            yield event
        return

    yield _event("error", {
        "message": "Unexpected session state. Please start a new conversation."
    })
    yield _event("done", {"session_id": session_id})


async def _stream_prediction(
    session: "ChatSession",
    confirmed: list[str],
    denied: list[str],
    summary: str,
    history: list[dict],
    db: "AsyncSession",
    session_id: str,
) -> AsyncGenerator[dict, None]:
    """Run Layer 2 then stream Layer 3 or the web search fallback.

    Args:
        session:    ChatSession ORM object (mutated in-place).
        confirmed:  Confirmed symptom tokens.
        denied:     Denied symptom tokens.
        summary:    Plain-English symptom summary.
        history:    Full conversation history to persist.
        db:         Async database session for persistence.
        session_id: Session identifier for log correlation.

    Yields:
        SSE event dicts.
    """
    from app.services import session_service

    yield _event("thinking", {"stage": "layer2", "message": "Analysing your symptoms..."})

    dt_result = None
    dt_failed = False

    try:
        dt_result = predict_disease(confirmed)
        if (
            not dt_result
            or dt_result.get("disease") is None
            or dt_result.get("confidence", 0) < 0.40
        ):
            dt_failed = True
            logger.info(
                "Layer 2 low confidence — routing to fallback | session=%s", session_id
            )
    except Exception as exc:
        dt_failed = True
        logger.warning(
            "Layer 2 failed: %s — routing to fallback | session=%s", exc, session_id
        )

    # ── Layer 3: streamed ────────────────────────────────────────────────────
    if not dt_failed:
        yield _event("thinking", {
            "stage": "layer3",
            "message": "Dr. Melvis is preparing your assessment...",
        })

        layer3_result = None

        async for event in _stream_layer3(
            dt_result["disease"],
            dt_result["confidence"],
            confirmed, denied, summary, session_id,
        ):
            if event["type"] == "token":
                yield event
            elif event["type"] == "_layer3_complete":
                layer3_result = event["data"]["result"]

        if layer3_result and layer3_result.get("is_plausible"):
            session.state = "DONE"
            session.completed = True
            session.confirmed_symptoms = confirmed
            session.conversation_history = history
            session.final_diagnosis = layer3_result
            session.predicted_disease = (
                layer3_result.get("display_disease") or dt_result["disease"]
            )
            session.confidence = (
                layer3_result.get("display_confidence") or dt_result["confidence"]
            )
            await session_service.save_session(db, session)
            yield _event("diagnosis", layer3_result)
            yield _event("done", {"session_id": session_id})
            return

        # Layer 3 implausible or failed — escalate to web search fallback.
        dt_failed = True
        logger.info(
            "Layer 3 implausible or failed — escalating to fallback | session=%s",
            session_id,
        )

    # ── Web search fallback: streamed ────────────────────────────────────────
    yield _event("thinking", {
        "stage": "searching",
        "message": "Dr. Melvis is researching your symptoms...",
    })

    fallback_result = None

    async for event in _stream_fallback(confirmed, denied, summary, session_id):
        if event["type"] == "token":
            yield event
        elif event["type"] == "_fallback_complete":
            fallback_result = event["data"]["result"]

    final_result = fallback_result or _inconclusive_fallback()

    session.state = "DONE"
    session.completed = True
    session.confirmed_symptoms = confirmed
    session.conversation_history = history
    session.final_diagnosis = final_result
    session.predicted_disease = final_result.get("display_disease")
    session.confidence = final_result.get("display_confidence")
    await session_service.save_session(db, session)
    yield _event("diagnosis", final_result)
    yield _event("done", {"session_id": session_id})


async def _stream_layer3(
    predicted_disease: str,
    raw_confidence: float,
    confirmed: list[str],
    denied: list[str],
    summary: str,
    session_id: str,
) -> AsyncGenerator[dict, None]:
    """Stream Layer 3 (output validator) using Claude's async streaming API.

    Yields ``token`` events as Claude streams the response, then a private
    ``_layer3_complete`` event containing the fully parsed result dict.

    Args:
        predicted_disease: Disease name from the Decision Tree.
        raw_confidence:    Raw predict_proba value from the DT.
        confirmed:         Confirmed symptom tokens.
        denied:            Denied symptom tokens.
        summary:           Plain-English symptom summary.
        session_id:        Session identifier for log correlation.

    Yields:
        ``token`` events (``{"token": str}``) followed by one
        ``_layer3_complete`` event (``{"result": dict | None}``).
    """
    display_confidence = dampen_confidence(raw_confidence)

    user_message = (
        f"Decision Tree prediction: {predicted_disease}\n"
        f"Raw confidence: {raw_confidence:.3f}\n"
        f"Display confidence: {display_confidence:.3f}\n"
        f"Confirmed symptoms: {confirmed}\n"
        f"Denied symptoms: {denied}\n"
        f"Conversation summary: {summary}\n\n"
        f"Validate this prediction and generate the response JSON."
    )

    full_text = ""

    try:
        async with _aclient.messages.stream(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=LAYER3_SYSTEM_PROMPT.format(disclaimer=MEDICAL_DISCLAIMER),
            messages=[{"role": "user", "content": user_message}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                yield _event("token", {"token": text})

        try:
            parsed = extract_json_from_response(full_text)
            if not parsed.get("disclaimer"):
                parsed["disclaimer"] = MEDICAL_DISCLAIMER
            if parsed.get("display_confidence", 0) > 0.89:
                parsed["display_confidence"] = 0.89
            yield _event("_layer3_complete", {"result": parsed})
        except ValueError as exc:
            logger.error(
                "Layer 3 stream JSON parse failed: %s | session=%s", exc, session_id
            )
            yield _event("_layer3_complete", {"result": None})

    except Exception as exc:
        logger.error("Layer 3 stream API error: %s | session=%s", exc, session_id)
        yield _event("_layer3_complete", {"result": None})


async def _stream_fallback(
    confirmed: list[str],
    denied: list[str],
    summary: str,
    session_id: str,
) -> AsyncGenerator[dict, None]:
    """Stream the web search fallback using Claude's async streaming API.

    Called when the Decision Tree cannot produce a reliable prediction.
    Uses Claude with web search to research the symptom cluster.

    Args:
        confirmed:  Confirmed symptom tokens.
        denied:     Denied symptom tokens.
        summary:    Plain-English symptom summary.
        session_id: Session identifier for log correlation.

    Yields:
        ``token`` events (``{"token": str}``) followed by one
        ``_fallback_complete`` event (``{"result": dict | None}``).
    """
    system = (
        "You are Dr. Melvis, a warm and knowledgeable AI health assistant.\n"
        "The patient's symptoms could not be matched to conditions in your local diagnostic\n"
        "database. Use web search to research the symptom cluster and provide a helpful,\n"
        "clinically grounded response.\n\n"
        "RULES:\n"
        "- Search for the most likely condition(s) given the symptoms\n"
        "- Be honest that this is an AI assessment, not a professional diagnosis\n"
        "- Give 3-5 practical precautions or next steps\n"
        "- Always include the disclaimer\n"
        "- Respond in warm, human, plain English — not clinical jargon\n"
        "- Format your FINAL answer as a JSON block:\n"
        "```json\n"
        "{\n"
        '  "is_plausible": true,\n'
        '  "display_disease": "Most likely condition name",\n'
        '  "display_confidence": 0.72,\n'
        '  "urgency": "mild | moderate | urgent",\n'
        '  "explanation": "Plain English explanation",\n'
        '  "precautions": ["step 1", "step 2", "step 3"],\n'
        '  "when_to_see_doctor": "Specific guidance",\n'
        '  "source": "web_search",\n'
        '  "disclaimer": "PASTE DISCLAIMER HERE"\n'
        "}\n"
        "```"
    )

    user_message = (
        f"Patient confirmed symptoms: {confirmed}\n"
        f"Patient denied symptoms: {denied}\n"
        f"Conversation summary: {summary}\n\n"
        f"The local diagnostic model could not handle this symptom cluster. "
        f"Please search for the most likely condition and provide your assessment."
    )

    full_text = ""

    try:
        async with _aclient.messages.stream(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                yield _event("token", {"token": text})

        try:
            parsed = extract_json_from_response(full_text)
            parsed["disclaimer"] = MEDICAL_DISCLAIMER
            parsed["source"] = "web_search"
            if parsed.get("display_confidence", 0) > 0.89:
                parsed["display_confidence"] = 0.89
            yield _event("_fallback_complete", {"result": parsed})
        except ValueError as exc:
            logger.error(
                "Fallback stream JSON parse failed: %s | session=%s", exc, session_id
            )
            yield _event("_fallback_complete", {"result": None})

    except Exception as exc:
        logger.error("Fallback stream API error: %s | session=%s", exc, session_id)
        yield _event("_fallback_complete", {"result": None})


def _inconclusive_fallback() -> dict:
    """Safe fallback dict when all diagnostic paths fail.

    Returns:
        Minimal inconclusive response with disclaimer and doctor referral.
    """
    return {
        "is_plausible": False,
        "display_disease": None,
        "display_confidence": None,
        "urgency": "moderate",
        "explanation": (
            "We were unable to determine a clear diagnosis from the symptoms described. "
            "This may be because your symptoms match multiple conditions or fall outside "
            "the conditions this system handles."
        ),
        "precautions": [
            "Visit the nearest clinic or hospital for a proper evaluation",
            "Describe all your symptoms to the doctor as you described them here",
            "Do not self-medicate based on this assessment",
        ],
        "when_to_see_doctor": "As soon as possible",
        "disclaimer": MEDICAL_DISCLAIMER,
    }
