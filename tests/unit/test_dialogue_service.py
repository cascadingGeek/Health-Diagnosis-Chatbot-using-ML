"""Unit tests for app/services/dialogue_service.py."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.dialogue_service import (
    DialogueState,
    TurnResult,
    _parse_yes_no,
    process_message,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_session(
    state: str = "GREETING",
    confirmed: list | None = None,
    asked: list | None = None,
    predicted_disease: str | None = None,
    confidence: float | None = None,
    completed: bool = False,
) -> MagicMock:
    """Create a mock ChatSession with the given state."""
    session = MagicMock()
    session.id = uuid.uuid4()
    session.state = state
    session.confirmed_symptoms = confirmed or []
    session.asked_symptoms = asked or []
    session.predicted_disease = predicted_disease
    session.confidence = confidence
    session.completed = completed
    return session


SYMPTOM_LIST_SHORT = [
    "itching", "skin_rash", "chills", "joint_pain",
    "fatigue", "fever", "headache", "nausea", "vomiting",
    "stomach_pain",
]


def _make_registry(symptom_list: list[str] | None = None) -> MagicMock:
    registry = MagicMock()
    registry.get_symptom_list.return_value = symptom_list or SYMPTOM_LIST_SHORT
    return registry


# ── _parse_yes_no ─────────────────────────────────────────────────────────────


class TestParseYesNo:
    @pytest.mark.parametrize("text", ["yes", "YES", "y", "yeah", "yep", "sure", "1"])
    def test_yes_variants(self, text):
        assert _parse_yes_no(text) is True

    @pytest.mark.parametrize("text", ["no", "NO", "n", "nope", "nah", "0"])
    def test_no_variants(self, text):
        assert _parse_yes_no(text) is False

    @pytest.mark.parametrize("text", ["maybe", "dunno", "what?", ""])
    def test_ambiguous_returns_none(self, text):
        assert _parse_yes_no(text) is None


# ── GREETING state ────────────────────────────────────────────────────────────


class TestGreetingState:
    def test_transitions_to_collecting(self):
        session = _make_session(state="GREETING")
        registry = _make_registry()
        result = process_message(session, "", registry)
        assert session.state == DialogueState.COLLECTING

    def test_returns_welcome_message(self):
        session = _make_session(state="GREETING")
        registry = _make_registry()
        result = process_message(session, "", registry)
        assert "hello" in result.bot_message.lower() or "symptom" in result.bot_message.lower()

    def test_no_diagnosis_in_greeting(self):
        session = _make_session(state="GREETING")
        registry = _make_registry()
        result = process_message(session, "", registry)
        assert result.diagnosis is None


# ── COLLECTING state ──────────────────────────────────────────────────────────


class TestCollectingState:
    def test_unrecognised_primary_symptom_prompts_retry(self):
        session = _make_session(state="COLLECTING")
        registry = _make_registry()
        result = process_message(session, "xyzzy_unknown_abc_987", registry)
        assert "couldn't recognise" in result.bot_message.lower() or \
               "rephrase" in result.bot_message.lower()
        assert session.confirmed_symptoms == []

    def test_recognised_primary_symptom_adds_to_confirmed(self):
        session = _make_session(state="COLLECTING")
        registry = _make_registry()
        result = process_message(session, "itching", registry)
        assert "itching" in session.confirmed_symptoms

    def test_yes_answer_adds_candidate_to_confirmed(self):
        # Session already has primary confirmed; first candidate has been asked
        session = _make_session(
            state="COLLECTING",
            confirmed=["itching"],
            asked=["itching", "skin_rash"],
        )
        registry = _make_registry()
        result = process_message(session, "yes", registry)
        assert "skin_rash" in session.confirmed_symptoms

    def test_no_answer_does_not_add_to_confirmed(self):
        session = _make_session(
            state="COLLECTING",
            confirmed=["itching"],
            asked=["itching", "skin_rash"],
        )
        registry = _make_registry()
        before = list(session.confirmed_symptoms)
        process_message(session, "no", registry)
        assert "skin_rash" not in session.confirmed_symptoms

    def test_ambiguous_answer_requests_clarification(self):
        session = _make_session(
            state="COLLECTING",
            confirmed=["itching"],
            asked=["itching", "skin_rash"],
        )
        registry = _make_registry()
        result = process_message(session, "maybe", registry)
        assert "yes" in result.bot_message.lower() or "no" in result.bot_message.lower()

    def test_transitions_to_confirming_when_done(self):
        # 3+ confirmed, enough questions asked
        all_syms = SYMPTOM_LIST_SHORT[:]
        confirmed = all_syms[:3]
        asked = all_syms  # all asked = no remaining
        session = _make_session(
            state="COLLECTING",
            confirmed=confirmed,
            asked=asked,
        )
        registry = _make_registry()
        process_message(session, "no", registry)
        assert session.state == DialogueState.CONFIRMING


# ── CONFIRMING state ──────────────────────────────────────────────────────────


class TestConfirmingState:
    def test_yes_transitions_to_done(self):
        session = _make_session(
            state="CONFIRMING",
            confirmed=["itching", "chills", "fever"],
        )
        registry = _make_registry()
        with patch("app.services.dialogue_service.diagnose") as mock_diagnose:
            from app.schemas.diagnosis import DiagnosisResult
            mock_diagnose.return_value = DiagnosisResult(
                disease="Flu", confidence=0.9,
                description="desc", precautions=[]
            )
            process_message(session, "yes", registry)
        # After confirming yes, state should be DONE (via PREDICTING)
        assert session.state == DialogueState.DONE

    def test_no_resets_and_returns_to_collecting(self):
        session = _make_session(
            state="CONFIRMING",
            confirmed=["itching", "chills", "fever"],
            asked=["itching", "chills", "fever", "nausea"],
        )
        registry = _make_registry()
        result = process_message(session, "no", registry)
        assert session.state == DialogueState.COLLECTING
        assert session.confirmed_symptoms == []
        assert session.asked_symptoms == []

    def test_ambiguous_prompts_clarification(self):
        session = _make_session(
            state="CONFIRMING",
            confirmed=["itching"],
        )
        registry = _make_registry()
        result = process_message(session, "hmm", registry)
        assert session.state == DialogueState.CONFIRMING
        assert "yes" in result.bot_message.lower() or "no" in result.bot_message.lower()


# ── DONE state ────────────────────────────────────────────────────────────────


class TestDoneState:
    def test_restart_hint(self):
        session = _make_session(state="DONE", completed=True)
        registry = _make_registry()
        result = process_message(session, "restart", registry)
        assert "session" in result.bot_message.lower() or "new" in result.bot_message.lower()

    def test_generic_done_response(self):
        session = _make_session(state="DONE", completed=True)
        registry = _make_registry()
        result = process_message(session, "thanks", registry)
        assert isinstance(result, TurnResult)
