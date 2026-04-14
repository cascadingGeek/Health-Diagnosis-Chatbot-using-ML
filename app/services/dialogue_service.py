"""Conversation state machine for the multi-turn diagnostic dialogue.

States
------
GREETING   → Bot sends welcome and asks for the user's primary symptom.
COLLECTING → Bot asks yes/no questions for candidate symptoms.
CONFIRMING → Bot lists confirmed symptoms and asks for user confirmation.
PREDICTING → Bot runs inference and returns the diagnosis.
DONE       → Bot offers restart / feedback.

All mutable state is stored in the ``ChatSession`` ORM object passed in from
the controller.  This service is fully stateless and never holds module-level
mutable data.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from rapidfuzz import process as rfprocess

from app.core.exceptions import FeatureContractError, InconclusiveDiagnosisError
from app.ml.feature_builder import _normalise  # reuse normalisation logic
from app.ml.model_registry import ModelRegistry
from app.schemas.diagnosis import DiagnosisResult

if TYPE_CHECKING:
    from app.database.models.session_log import ChatSession

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum confirmed symptoms before collection can stop.
_MIN_CONFIRMED: int = 3
# Maximum yes/no questions asked per session.
_MAX_QUESTIONS: int = 10
# Fuzzy match threshold for the user's free-text primary symptom input.
_PRIMARY_FUZZY_THRESHOLD: float = 60.0


class DialogueState(StrEnum):
    """Valid states of the diagnosis dialogue."""

    GREETING = "GREETING"
    COLLECTING = "COLLECTING"
    CONFIRMING = "CONFIRMING"
    PREDICTING = "PREDICTING"
    DONE = "DONE"


# ── State-handler return type ─────────────────────────────────────────────────


class TurnResult:
    """Container for the outcome of processing a single dialogue turn.

    Attributes:
        bot_message: Text to return to the user.
        diagnosis:   Populated only when the state transitions to PREDICTING.
    """

    def __init__(
        self,
        bot_message: str,
        diagnosis: DiagnosisResult | None = None,
    ) -> None:
        self.bot_message = bot_message
        self.diagnosis = diagnosis


# ── Public entry point ────────────────────────────────────────────────────────


def process_message(
    session: "ChatSession",
    user_message: str,
    registry: ModelRegistry,
) -> TurnResult:
    """Advance the dialogue state machine by one turn.

    Mutates *session* in-place.  The caller (controller) is responsible for
    persisting the updated session to the database.

    Args:
        session:      The current ``ChatSession`` ORM object.
        user_message: Raw text sent by the user.
        registry:     Initialised ``ModelRegistry`` (needed for PREDICTING).

    Returns:
        A ``TurnResult`` with the bot's reply and an optional diagnosis.
    """
    state = DialogueState(session.state)

    if state == DialogueState.GREETING:
        return _handle_greeting(session)

    if state == DialogueState.COLLECTING:
        return _handle_collecting(session, user_message, registry)

    if state == DialogueState.CONFIRMING:
        return _handle_confirming(session, user_message, registry)

    if state == DialogueState.PREDICTING:
        return _handle_predicting(session, registry)

    if state == DialogueState.DONE:
        return _handle_done(session, user_message)

    # Should never be reached given the enum, but keeps type-checker happy.
    return TurnResult("I'm not sure what to do. Let's start over.", None)  # pragma: no cover


# ── State handlers ─────────────────────────────────────────────────────────────


def _handle_greeting(session: "ChatSession") -> TurnResult:
    """Emit welcome message and transition to COLLECTING.

    Args:
        session: Current session (mutated in-place).

    Returns:
        TurnResult with the welcome message.
    """
    session.state = DialogueState.COLLECTING
    return TurnResult(
        "Hello! I'm Dr. Melvis, your AI health assistant. "
        "I'll ask you a few questions about your symptoms to help identify "
        "a possible condition.\n\n"
        "Please describe your **primary symptom** (e.g. 'fever', 'headache', "
        "'joint pain')."
    )


def _handle_collecting(
    session: "ChatSession",
    user_message: str,
    registry: ModelRegistry,
) -> TurnResult:
    """Process a user reply in the COLLECTING state.

    If no symptoms have been confirmed yet, treat the message as the user's
    primary symptom (free-text → fuzzy match).  Otherwise treat it as a
    yes/no answer to the last question.

    Args:
        session:      Current session (mutated in-place).
        user_message: User's raw text.
        registry:     Used to obtain the canonical symptom list.

    Returns:
        TurnResult with the next yes/no question or confirmation prompt.
    """
    symptom_list = registry.get_symptom_list()
    confirmed: list[str] = list(session.confirmed_symptoms)
    asked: list[str] = list(session.asked_symptoms)

    # ── Branch A: no confirmed symptoms yet — parse primary symptom ──────────
    if not confirmed:
        matched = _fuzzy_match_primary(user_message, symptom_list)
        if matched is None:
            return TurnResult(
                f"I couldn't recognise '{user_message}' as a known symptom. "
                "Could you rephrase it? For example: 'fever', 'cough', "
                "'chest pain', 'fatigue'."
            )
        confirmed.append(matched)
        asked.append(matched)
        session.confirmed_symptoms = confirmed
        session.asked_symptoms = asked

        # Ask the first candidate symptom right away.
        return _ask_next_or_confirm(session, symptom_list, confirmed, asked)

    # ── Branch B: parse yes/no answer ─────────────────────────────────────────
    answer = _parse_yes_no(user_message)
    if answer is None:
        # Last asked symptom is the one awaiting a yes/no.
        last_asked = asked[-1] if asked else "this symptom"
        return TurnResult(
            f"Sorry, I didn't catch that. "
            f"Do you have **{last_asked.replace('_', ' ')}**? "
            "Please answer **yes** or **no**."
        )

    if answer is True:
        # The last asked symptom (which isn't the primary) was confirmed.
        # The last entry of `asked` is the symptom we just asked about.
        last_candidate = asked[-1]
        if last_candidate not in confirmed:
            confirmed.append(last_candidate)
            session.confirmed_symptoms = confirmed

    session.asked_symptoms = asked  # already up-to-date

    return _ask_next_or_confirm(session, symptom_list, confirmed, asked)


def _ask_next_or_confirm(
    session: "ChatSession",
    symptom_list: list[str],
    confirmed: list[str],
    asked: list[str],
) -> TurnResult:
    """Either ask the next candidate symptom or transition to CONFIRMING.

    Args:
        session:      Current session (mutated in-place).
        symptom_list: Full canonical symptom list.
        confirmed:    Symptoms confirmed so far.
        asked:        Symptoms already presented to the user.

    Returns:
        TurnResult with the next question or the confirmation prompt.
    """
    questions_asked = len(asked) - 1  # subtract the primary symptom
    remaining = [s for s in symptom_list if s not in asked]

    enough_confirmed = len(confirmed) >= _MIN_CONFIRMED
    no_more_questions = not remaining or questions_asked >= _MAX_QUESTIONS

    if enough_confirmed and no_more_questions:
        session.state = DialogueState.CONFIRMING
        names = ", ".join(s.replace("_", " ") for s in confirmed)
        return TurnResult(
            f"Based on your responses I've noted the following symptoms:\n"
            f"**{names}**\n\n"
            "Is that correct? (yes / no)"
        )

    if not remaining:
        # Ran out of candidates without 3 confirmed — proceed anyway.
        session.state = DialogueState.CONFIRMING
        names = ", ".join(s.replace("_", " ") for s in confirmed) or "none"
        return TurnResult(
            f"I've gone through all available symptoms. You confirmed:\n"
            f"**{names}**\n\nIs that correct? (yes / no)"
        )

    # Pick the next candidate and add it to asked.
    next_symptom = remaining[0]
    asked.append(next_symptom)
    session.asked_symptoms = asked

    readable = next_symptom.replace("_", " ")
    return TurnResult(f"Do you have **{readable}**? (yes / no)")


def _handle_confirming(
    session: "ChatSession",
    user_message: str,
    registry: ModelRegistry,
) -> TurnResult:
    """Process yes/no confirmation of the collected symptom list.

    yes → PREDICTING (run inference immediately).
    no  → reset and return to COLLECTING.

    Args:
        session:      Current session (mutated in-place).
        user_message: User's response to the confirmation prompt.
        registry:     Passed through for the PREDICTING branch.

    Returns:
        TurnResult with diagnosis or reset message.
    """
    answer = _parse_yes_no(user_message)
    if answer is None:
        names = ", ".join(
            s.replace("_", " ") for s in session.confirmed_symptoms
        )
        return TurnResult(
            f"Sorry, please answer **yes** or **no**.\n"
            f"Confirmed symptoms: **{names}**"
        )

    if answer is False:
        # Reset collection
        session.confirmed_symptoms = []
        session.asked_symptoms = []
        session.state = DialogueState.COLLECTING
        return TurnResult(
            "No problem! Let's start again. "
            "Please describe your **primary symptom**."
        )

    # answer is True → run inference
    session.state = DialogueState.PREDICTING
    return _handle_predicting(session, registry)


def _handle_predicting(
    session: "ChatSession",
    registry: ModelRegistry,
) -> TurnResult:
    """Run the ML inference pipeline and store the result on the session.

    Args:
        session:  Current session (mutated in-place).
        registry: Initialised ``ModelRegistry``.

    Returns:
        TurnResult with disease name, confidence, and advice — or an
        inconclusive message if confidence is too low.
    """
    from app.services.diagnosis_service import diagnose

    session.state = DialogueState.DONE
    session.completed = True

    try:
        result = diagnose(list(session.confirmed_symptoms), registry)
    except FeatureContractError as exc:
        logger.warning("Feature contract error during prediction", extra={"error": str(exc)})
        return TurnResult(
            "I encountered an unrecognised symptom and cannot complete the "
            "diagnosis. Please restart and try again."
        )
    except InconclusiveDiagnosisError:
        return TurnResult(
            "Based on the symptoms provided, the results are **inconclusive**. "
            "Please consult a qualified medical professional for a proper evaluation."
        )

    session.predicted_disease = result.disease
    session.confidence = result.confidence

    prec_text = (
        "\n".join(f"- {p}" for p in result.precautions)
        if result.precautions
        else "No specific precautions available."
    )

    message = (
        f"Based on your symptoms, the most likely condition is:\n\n"
        f"**{result.disease}** (confidence: {result.confidence:.0%})\n\n"
        f"{result.description}\n\n"
        f"**Recommended precautions:**\n{prec_text}\n\n"
        f"*This is not a substitute for professional medical advice.*\n\n"
        f"Would you like to give feedback (rate 1-5) or start a new session? "
        f"Type **restart** to begin again."
    )
    return TurnResult(message, result)


def _handle_done(session: "ChatSession", user_message: str) -> TurnResult:
    """Handle messages in the DONE state (restart or feedback prompt).

    Args:
        session:      Current session.
        user_message: User's text (checked for 'restart' keyword).

    Returns:
        TurnResult guiding the user to restart or submit feedback.
    """
    if "restart" in user_message.lower():
        # Controller will create a new session; instruct user.
        return TurnResult(
            "To start a new session, simply send a new message without a "
            "session_id, or refresh the page."
        )
    return TurnResult(
        "This session is complete. "
        "You can submit feedback via the /api/v1/feedback endpoint, "
        "or start a new conversation by omitting the session_id."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fuzzy_match_primary(raw: str, symptom_list: list[str]) -> str | None:
    """Fuzzy-match user free-text to the closest canonical symptom.

    Args:
        raw:          User-entered primary symptom text.
        symptom_list: Canonical symptom list.

    Returns:
        Matched canonical symptom name, or ``None`` if no good match found.
    """
    normalised_input = _normalise(raw)
    normalised_list = [_normalise(s) for s in symptom_list]
    result = rfprocess.extractOne(
        normalised_input,
        normalised_list,
        score_cutoff=_PRIMARY_FUZZY_THRESHOLD,
    )
    if result is None:
        return None
    _, _, idx = result
    return symptom_list[idx]


def _parse_yes_no(text: str) -> bool | None:
    """Parse a free-text response into True (yes), False (no), or None.

    Args:
        text: User's raw response string.

    Returns:
        ``True`` for yes-like responses, ``False`` for no-like responses,
        ``None`` if the intent is ambiguous.
    """
    lowered = text.strip().lower()
    if lowered in {"yes", "y", "yeah", "yep", "yup", "sure", "correct", "true", "1"}:
        return True
    if lowered in {"no", "n", "nope", "nah", "false", "0"}:
        return False
    return None
