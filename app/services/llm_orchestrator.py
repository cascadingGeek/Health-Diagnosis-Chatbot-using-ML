"""
Orchestrates the three-layer hybrid diagnostic pipeline.

Flow:
  User message
      ↓
  Layer 1 — Claude (llm_input_processor)
      Understands free text, extracts symptoms, generates follow-up questions.
      Does NOT predict diseases.
      ↓
  Layer 2 — Decision Tree (predictor_service)
      Maps binary symptom vector to disease prediction + confidence score.
      This is the ONLY component that makes diagnostic predictions.
      ↓
  Layer 3 — Claude (llm_output_validator)
      Validates clinical plausibility, generates human-readable explanation,
      adds precautions and medical disclaimer.
      ↓
  Response to user

This module is the single entry point called by the chat controller for every
user message once the FSM is in COLLECTING or CONFIRMING state.
"""

import logging
from typing import TYPE_CHECKING

from anthropic import Anthropic

from app.core.config import settings
from app.core.json_utils import extract_json_from_response
from app.core.llm_input_processor import extract_symptoms_from_text
from app.core.llm_output_validator import MEDICAL_DISCLAIMER, validate_and_explain_diagnosis
from app.services.predictor_service import predict_disease

_client = Anthropic(api_key=settings.anthropic_api_key)

if TYPE_CHECKING:
    from app.database.models.session_log import ChatSession

logger = logging.getLogger(__name__)


async def process_message(
    session: "ChatSession",
    user_message: str,
) -> dict:
    """Main pipeline entry point.  Called by the chat controller for every message.

    Mutates *session* in-place (state, confirmed_symptoms, denied_symptoms,
    conversation_history, questions_asked, final_diagnosis).  The controller is
    responsible for persisting the updated session to the database.

    Args:
        session: The active ChatSession ORM object (already fetched by the controller).
        user_message: Raw text sent by the user.

    Returns:
        Dict with keys:
          - response_type: "question" | "diagnosis" | "inconclusive" | "error"
          - message: Text to display to the user.
          - diagnosis: Full Layer 3 result dict (only when response_type is
            "diagnosis" or "inconclusive").
          - session_state: The FSM state string after this turn.
    """
    session_id = str(session.id)
    state = session.state
    confirmed: list[str] = list(session.confirmed_symptoms or [])
    denied: list[str] = list(session.denied_symptoms or [])
    history: list[dict] = list(session.conversation_history or [])
    questions_asked: int = session.questions_asked or 0

    # Append the user's message to conversation history before processing.
    history.append({"role": "user", "content": user_message})

    # ── COLLECTING / GREETING STATE ──────────────────────────────────────────
    if state in ("GREETING", "COLLECTING"):

        try:
            layer1_result = extract_symptoms_from_text(
                user_message=user_message,
                conversation_history=history[:-1],  # exclude the just-appended message
                confirmed_so_far=confirmed,
                denied_so_far=denied,
                questions_asked=questions_asked,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error(
                "Layer 1 call failed: %s | session=%s", exc, session_id
            )
            return _error_response(
                "I had trouble understanding your message. "
                "Could you describe your symptoms again?",
                session,
            )

        action = layer1_result.get("action")

        # Merge newly extracted symptoms into session lists (no duplicates).
        for symptom in layer1_result.get("confirmed_symptoms", []):
            if symptom and symptom not in confirmed:
                confirmed.append(symptom)
        for symptom in layer1_result.get("denied_symptoms", []):
            if symptom and symptom not in denied:
                denied.append(symptom)

        if action == "ask":
            question = layer1_result.get(
                "question",
                "Could you tell me more about how you are feeling?",
            )
            # Capture symptoms resolved this turn via severity/qualifier statements.
            # These come from newly_confirmed/newly_denied in the ask action, which
            # is separate from the generic confirmed_symptoms/denied_symptoms already
            # merged above (those apply to extract/ready actions).
            for s in layer1_result.get("newly_confirmed", []):
                if s and s not in confirmed:
                    confirmed.append(s)
            for s in layer1_result.get("newly_denied", []):
                if s and s not in denied:
                    denied.append(s)

            history.append({"role": "assistant", "content": question})
            _update_session_collecting(
                session, confirmed, denied, history, questions_asked + 1
            )
            return {
                "response_type": "question",
                "message": question,
                "diagnosis": None,
                "session_state": "COLLECTING",
            }

        if action in ("ready", "extract") and len(confirmed) >= 3:
            # Enough symptoms — proceed to prediction.
            summary = layer1_result.get(
                "summary", f"Symptoms reported: {', '.join(confirmed)}"
            )
            _update_session_collecting(
                session, confirmed, denied, history, questions_asked
            )
            return await _run_prediction(session, confirmed, denied, summary, history)

        if action == "extract" and len(confirmed) < 3:
            # Layer 1 extracted symptoms but did not generate a follow-up question.
            # Use unmapped_complaints for context when available.
            unmapped = layer1_result.get("unmapped_complaints", [])
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
            _update_session_collecting(
                session, confirmed, denied, history, questions_asked + 1
            )
            return {
                "response_type": "question",
                "message": follow_up,
                "diagnosis": None,
                "session_state": "COLLECTING",
            }

        # ── UNCLEAR — Layer 1 could not parse the user's message ───────────────
        # Check whether we already asked for clarification on the previous turn.
        # If so, do not loop — forward to prediction with whatever is confirmed.
        last_assistant = next(
            (m["content"] for m in reversed(history[:-1]) if m["role"] == "assistant"),
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
                "UNCLEAR loop broken — proceeding to prediction | "
                "confirmed=%s questions_asked=%d | session=%s",
                confirmed, questions_asked, session_id,
            )
            summary = (
                f"Patient described: "
                f"{', '.join(confirmed) if confirmed else 'unspecified symptoms'}. "
                f"Some responses could not be fully interpreted."
            )
            _update_session_collecting(
                session, confirmed, denied, history, questions_asked
            )
            return await _run_prediction(session, confirmed, denied, summary, history)

        # First UNCLEAR this session — ask politely once.
        follow_up = (
            "I want to make sure I understand you correctly. "
            "Could you tell me a little more about how you are feeling?"
        )
        history.append({"role": "assistant", "content": follow_up})
        _update_session_collecting(
            session, confirmed, denied, history, questions_asked + 1
        )
        return {
            "response_type": "question",
            "message": follow_up,
            "diagnosis": None,
            "session_state": "COLLECTING",
        }

    # ── CONFIRMING STATE ─────────────────────────────────────────────────────
    if state == "CONFIRMING":
        summary = f"Confirmed symptoms: {', '.join(confirmed)}"
        return await _run_prediction(session, confirmed, denied, summary, history)

    return _error_response(
        "Unexpected session state. Please start a new conversation.", session
    )


async def _run_prediction(
    session: "ChatSession",
    confirmed: list[str],
    denied: list[str],
    summary: str,
    history: list[dict],
) -> dict:
    """Run Layer 2 (Decision Tree) then Layer 3 (Claude output validator).

    Falls back to Claude web search if the Decision Tree cannot produce a
    reliable prediction (sparse symptom vector, unknown symptoms, or low
    confidence).

    Args:
        session: Active ChatSession (mutated in-place).
        confirmed: Confirmed symptom tokens.
        denied: Denied symptom tokens.
        summary: Plain-English symptom summary for Layer 3 context.
        history: Full conversation history to persist.

    Returns:
        Final response dict with response_type "diagnosis" or "inconclusive".
    """
    session_id = str(session.id)

    # ── Attempt Layer 2: Decision Tree ──────────────────────────────────────
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
                "Layer 2 confidence too low or no result — routing to LLM fallback. "
                "Confirmed symptoms: %s", confirmed
            )
    except Exception as exc:
        dt_failed = True
        logger.warning(
            "Layer 2 prediction failed: %s — routing to LLM fallback | session=%s",
            exc, session_id,
        )

    # ── Layer 2 succeeded — run Layer 3 validation ──────────────────────────
    if not dt_failed:
        layer3_result = validate_and_explain_diagnosis(
            predicted_disease=dt_result["disease"],
            raw_confidence=dt_result["confidence"],
            confirmed_symptoms=confirmed,
            denied_symptoms=denied,
            conversation_summary=summary,
            session_id=session_id,
        )
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

        if layer3_result.get("is_plausible"):
            return {
                "response_type": "diagnosis",
                "message": layer3_result["explanation"],
                "diagnosis": layer3_result,
                "session_state": "DONE",
            }
        # Layer 3 flagged the prediction as implausible — escalate to LLM fallback.
        dt_failed = True
        logger.info(
            "Layer 3 flagged implausible — escalating to LLM fallback | session=%s",
            session_id,
        )

    # ── LLM Fallback: Claude with web search ────────────────────────────────
    logger.info(
        "Running LLM web search fallback for symptoms: %s | session=%s",
        confirmed, session_id,
    )
    fallback_result = await _llm_web_search_fallback(
        confirmed_symptoms=confirmed,
        denied_symptoms=denied,
        conversation_summary=summary,
        session_id=session_id,
    )
    session.state = "DONE"
    session.completed = True
    session.confirmed_symptoms = confirmed
    session.conversation_history = history
    session.final_diagnosis = fallback_result
    session.predicted_disease = fallback_result.get("display_disease")
    session.confidence = fallback_result.get("display_confidence")

    return {
        "response_type": "diagnosis",
        "message": fallback_result["explanation"],
        "diagnosis": fallback_result,
        "session_state": "DONE",
    }


def _update_session_collecting(
    session: "ChatSession",
    confirmed: list[str],
    denied: list[str],
    history: list[dict],
    questions_asked: int,
) -> None:
    """Write COLLECTING-state fields back onto the session object.

    Args:
        session: ChatSession ORM object to mutate.
        confirmed: Updated confirmed symptom list.
        denied: Updated denied symptom list.
        history: Updated conversation history.
        questions_asked: Updated question counter.
    """
    session.state = "COLLECTING"
    session.confirmed_symptoms = confirmed
    session.denied_symptoms = denied
    session.conversation_history = history
    session.questions_asked = questions_asked


def _error_response(message: str, session: "ChatSession") -> dict:
    """Build a safe error response dict.

    Args:
        message: Human-readable error text to display.
        session: Session object (state left unchanged).

    Returns:
        Response dict with response_type "error".
    """
    return {
        "response_type": "error",
        "message": message,
        "diagnosis": None,
        "session_state": session.state,
    }


async def _llm_web_search_fallback(
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    conversation_summary: str,
    session_id: str = "",
) -> dict:
    """Call Claude with web search when the Decision Tree cannot handle the symptom cluster.

    Used when the DT raises an exception, returns no result, produces confidence
    below the 0.40 floor, or when Layer 3 flags the prediction as implausible.
    Handles symptom clusters outside the 132-feature Kaggle vocabulary, sparse
    vectors, and conditions not in the 41 training classes.

    Args:
        confirmed_symptoms: Symptom tokens the user confirmed.
        denied_symptoms: Symptom tokens the user denied.
        conversation_summary: Plain-English summary of the consultation.
        session_id: Optional session identifier for log correlation.

    Returns:
        Diagnosis dict compatible with the Layer 3 output schema, with
        ``source`` set to ``"web_search"``.
    """
    system = (
        "You are Dr. Melvis, a warm and knowledgeable AI health assistant.\n"
        "The patient's symptoms could not be matched to conditions in your local diagnostic\n"
        "database. Use web search to research the symptom cluster and provide a helpful,\n"
        "clinically grounded response.\n\n"
        "RULES:\n"
        "- Search for the most likely condition(s) given the symptoms\n"
        "- Be honest that this is an AI assessment, not a professional diagnosis\n"
        "- Give 3–5 practical precautions or next steps\n"
        "- Always include the disclaimer\n"
        "- Respond in warm, human, plain English — not clinical jargon\n"
        "- Format your final answer as JSON:\n"
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
        "}"
    )

    user_message = (
        f"Patient confirmed symptoms: {confirmed_symptoms}\n"
        f"Patient denied symptoms: {denied_symptoms}\n"
        f"Conversation summary: {conversation_summary}\n\n"
        f"The local diagnostic model could not handle this symptom cluster. "
        f"Please search for the most likely condition and provide your assessment."
    )

    tools = [{"type": "web_search_20250305", "name": "web_search"}]

    raw = ""
    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=tools,
        )

        logger.info(
            "Claude API usage | model=%s input_tokens=%d output_tokens=%d layer=fallback",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            logger.warning("LLM web search fallback returned no text block")
            return _inconclusive_fallback()

        raw = text_blocks[-1].text.strip()
        try:
            parsed = extract_json_from_response(raw)
        except ValueError as exc:
            logger.error(
                "LLM web search fallback JSON extraction failed | error=%s | "
                "raw_length=%d | raw_preview=%s | session=%s",
                exc, len(raw), raw[:300], session_id,
            )
            return _inconclusive_fallback()
        parsed["disclaimer"] = MEDICAL_DISCLAIMER
        parsed["source"] = "web_search"
        # Enforce the 0.89 confidence ceiling.
        if parsed.get("display_confidence", 0) > 0.89:
            parsed["display_confidence"] = 0.89
        return parsed

    except Exception as exc:
        logger.error(
            "LLM web search fallback API error: %s | session=%s", exc, session_id
        )
        return _inconclusive_fallback()


def _inconclusive_fallback() -> dict:
    """Safe fallback diagnosis dict when no layer can produce a result.

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
        "source": "inconclusive",
        "disclaimer": MEDICAL_DISCLAIMER,
    }
