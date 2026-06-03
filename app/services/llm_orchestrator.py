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

from app.core.exceptions import FeatureContractError
from app.core.llm_input_processor import extract_symptoms_from_text
from app.core.llm_output_validator import validate_and_explain_diagnosis
from app.services.predictor_service import predict_disease

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
            # More symptoms needed — generate a bridging question.
            follow_up = (
                "Thank you for sharing that. Could you tell me more? "
                "For example, do you have a fever, headache, or any pain anywhere?"
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

        # Unclear / unrecognised input.
        follow_up = (
            "I didn't quite catch that. Could you describe what you're feeling "
            "in a bit more detail? You can type a full sentence."
        )
        history.append({"role": "assistant", "content": follow_up})
        session.conversation_history = history
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

    # ── LAYER 2: Decision Tree ───────────────────────────────────────────────
    try:
        dt_result = predict_disease(confirmed)
        predicted_disease = dt_result["disease"]
        raw_confidence = dt_result["confidence"]
    except FeatureContractError as exc:
        logger.error(
            "Layer 2 feature contract error: %s | session=%s", exc, session_id
        )
        return _error_response(
            "I encountered an unrecognised symptom and cannot complete the "
            "diagnosis. Please restart and try again.",
            session,
        )
    except Exception as exc:
        logger.error("Layer 2 prediction failed: %s | session=%s", exc, session_id)
        return _error_response(
            "The diagnostic model encountered an error. Please try again.", session
        )

    # ── LAYER 3: Claude validation + explanation ─────────────────────────────
    layer3_result = validate_and_explain_diagnosis(
        predicted_disease=predicted_disease,
        raw_confidence=raw_confidence,
        confirmed_symptoms=confirmed,
        denied_symptoms=denied,
        conversation_summary=summary,
        session_id=session_id,
    )

    # Persist final state on the session object.
    session.state = "DONE"
    session.completed = True
    session.confirmed_symptoms = confirmed
    session.conversation_history = history
    session.final_diagnosis = layer3_result
    session.predicted_disease = layer3_result.get("display_disease") or predicted_disease
    session.confidence = layer3_result.get("display_confidence") or raw_confidence

    if layer3_result.get("is_plausible"):
        return {
            "response_type": "diagnosis",
            "message": layer3_result["explanation"],
            "diagnosis": layer3_result,
            "session_state": "DONE",
        }
    return {
        "response_type": "inconclusive",
        "message": layer3_result["explanation"],
        "diagnosis": layer3_result,
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
