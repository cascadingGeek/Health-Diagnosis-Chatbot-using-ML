"""Symptom family routing — maps a primary symptom to clinically relevant follow-up questions.

Given a user's first-reported symptom, returns an ordered list of follow-up
symptom tokens to ask about, prioritised by clinical relevance for that
symptom family.  This replaces the original approach of iterating the full
132-symptom list in training-data order.

If the primary symptom is not recognised in any family, the module falls back
to the model's global ``feature_importances_`` ranking so the most
discriminative symptoms are always asked first.

All symptom tokens used in ``SYMPTOM_FAMILIES`` are chosen from the Kaggle
132-feature vocabulary and are resolvable by the feature builder's fuzzy
matcher even when the canonical CSV name contains an embedded space.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from rapidfuzz import process as rfprocess

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD: float = 70.0
_MAX_FOLLOWUPS: int = 8

# ---------------------------------------------------------------------------
# Symptom family definitions
# ---------------------------------------------------------------------------
# Keys are primary symptom tokens; values are ordered lists of follow-up
# symptom tokens (most clinically relevant first).  Every token must fuzzy-
# match to a canonical name in Training.csv via the feature builder.
# ---------------------------------------------------------------------------

SYMPTOM_FAMILIES: dict[str, list[str]] = {
    # ── HEAD / NEUROLOGICAL ────────────────────────────────────────────────
    "headache": [
        "nausea",
        "vomiting",
        "high_fever",
        "stiff_neck",
        "pain_behind_the_eyes",
        "blurred_and_distorted_vision",
        "dizziness",
        "fatigue",
        "loss_of_balance",
    ],
    # ── FEVER / SYSTEMIC INFECTION ────────────────────────────────────────
    "high_fever": [
        "chills",
        "sweating",
        "headache",
        "nausea",
        "vomiting",
        "muscle_pain",
        "fatigue",
        "loss_of_appetite",
        "yellowish_skin",
    ],
    # ── MALARIA-SPECIFIC ──────────────────────────────────────────────────
    "chills": [
        "high_fever",
        "sweating",
        "headache",
        "nausea",
        "vomiting",
        "muscle_pain",
        "fatigue",
        "diarrhoea",
    ],
    # ── TYPHOID-SPECIFIC ──────────────────────────────────────────────────
    "abdominal_pain": [
        "high_fever",
        "nausea",
        "vomiting",
        "diarrhoea",
        "constipation",
        "loss_of_appetite",
        "fatigue",
        "headache",
        "yellowish_skin",
    ],
    # Alias — stomach_pain and belly_pain route to the same family.
    "stomach_pain": [
        "high_fever",
        "nausea",
        "vomiting",
        "diarrhoea",
        "constipation",
        "loss_of_appetite",
        "fatigue",
        "headache",
        "yellowish_skin",
    ],
    "belly_pain": [
        "high_fever",
        "nausea",
        "vomiting",
        "diarrhoea",
        "constipation",
        "loss_of_appetite",
        "fatigue",
        "headache",
        "yellowish_skin",
    ],
    # ── STOMACH / GASTROINTESTINAL ────────────────────────────────────────
    "nausea": [
        "vomiting",
        "abdominal_pain",
        "diarrhoea",
        "loss_of_appetite",
        "indigestion",
        "stomach_pain",
        "fatigue",
        "high_fever",
    ],
    "vomiting": [
        "nausea",
        "abdominal_pain",
        "diarrhoea",
        "dehydration",
        "high_fever",
        "loss_of_appetite",
    ],
    "diarrhoea": [
        "vomiting",
        "stomach_pain",
        "abdominal_pain",
        "nausea",
        "dehydration",
        "high_fever",
        "fatigue",
    ],
    # ── JOINT / MUSCULOSKELETAL ───────────────────────────────────────────
    "joint_pain": [
        "swelling_joints",
        "muscle_pain",
        "stiff_neck",
        "fatigue",
        "skin_rash",
        "high_fever",
        "painful_walking",
        "back_pain",
    ],
    "muscle_pain": [
        "joint_pain",
        "fatigue",
        "high_fever",
        "chills",
        "back_pain",
        "stiff_neck",
    ],
    "back_pain": [
        "joint_pain",
        "muscle_pain",
        "fatigue",
        "constipation",
        "abdominal_pain",
        "stiff_neck",
    ],
    # ── SKIN ──────────────────────────────────────────────────────────────
    "skin_rash": [
        "itching",
        "joint_pain",
        "high_fever",
        "swelling_joints",
        "yellowish_skin",
        "blister",
        "skin_peeling",
    ],
    "itching": [
        "skin_rash",
        "yellowish_skin",
        "fatigue",
        "loss_of_appetite",
        "abdominal_pain",
    ],
    # ── RESPIRATORY ───────────────────────────────────────────────────────
    "cough": [
        "breathlessness",
        "phlegm",
        "high_fever",
        "chest_pain",
        "fatigue",
        "chills",
        "sweating",
        "loss_of_appetite",
        "throat_irritation",
    ],
    "breathlessness": [
        "cough",
        "chest_pain",
        "fatigue",
        "high_fever",
        "phlegm",
        "fast_heart_rate",
    ],
    # ── LIVER / JAUNDICE ──────────────────────────────────────────────────
    "yellowish_skin": [
        "dark_urine",
        "abdominal_pain",
        "fatigue",
        "loss_of_appetite",
        "nausea",
        "vomiting",
        "yellowing_of_eyes",
    ],
    # ── URINARY ───────────────────────────────────────────────────────────
    "burning_micturition": [
        "spotting_urination",
        "continuous_feel_of_urine",
        "bladder_discomfort",
        "fatigue",
        "high_fever",
        "foul_smell_of_urine",
    ],
    # ── FATIGUE / GENERAL ─────────────────────────────────────────────────
    "fatigue": [
        "high_fever",
        "loss_of_appetite",
        "weight_loss",
        "muscle_pain",
        "joint_pain",
        "nausea",
        "chills",
        "sweating",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    """Lowercase and replace whitespace / hyphens with underscores.

    Args:
        name: Raw symptom string from user input or the families dict.

    Returns:
        Normalised string suitable for fuzzy comparison.
    """
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _match_family_key(primary: str) -> str | None:
    """Fuzzy-match a primary symptom token to the closest SYMPTOM_FAMILIES key.

    Args:
        primary: Primary symptom token (may be canonical or user-supplied text).

    Returns:
        The matched family key string, or ``None`` if no match meets the
        confidence threshold.
    """
    keys = list(SYMPTOM_FAMILIES.keys())
    result = rfprocess.extractOne(
        _normalise(primary),
        keys,
        score_cutoff=_FUZZY_THRESHOLD,
    )
    if result is None:
        return None
    _, _, idx = result
    matched_key = keys[idx]
    logger.debug(
        "Family key matched",
        extra={"primary": primary, "family": matched_key},
    )
    return matched_key


def _fallback_from_importances(
    model: "DecisionTreeClassifier",
    symptom_list: list[str],
    exclude: set[str],
    limit: int,
) -> list[str]:
    """Return top-N symptoms by model feature importance, excluding *exclude*.

    Args:
        model:        Fitted ``DecisionTreeClassifier`` with ``feature_importances_``.
        symptom_list: Canonical 132-symptom list (same order as training features).
        exclude:      Symptom tokens to skip (already asked / the primary).
        limit:        Maximum number of symptoms to return.

    Returns:
        Ordered list of symptom tokens, most important first.
    """
    importances: np.ndarray = model.feature_importances_
    top_indices = np.argsort(importances)[::-1]
    result: list[str] = []
    for idx in top_indices:
        symptom = symptom_list[idx]
        if symptom not in exclude and importances[idx] > 0:
            result.append(symptom)
            if len(result) >= limit:
                break
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_followup_questions(
    primary_symptom: str,
    already_asked: list[str],
    symptom_list: list[str],
    model: "DecisionTreeClassifier | None" = None,
    max_questions: int = _MAX_FOLLOWUPS,
) -> list[str]:
    """Return an ordered list of follow-up symptom tokens to ask about.

    Looks up *primary_symptom* in ``SYMPTOM_FAMILIES`` to obtain a clinically
    ordered candidate list.  Any symptom already in *already_asked* is
    filtered out.  The result is capped at *max_questions*.

    Falls back to the model's ``feature_importances_`` when the primary
    symptom is not recognised.  If no model is provided either, returns
    symptom_list entries not yet asked, in training-data order.

    Args:
        primary_symptom: Canonical primary symptom token (e.g. ``"headache"``).
        already_asked:   Symptom tokens already presented to the user this session.
        symptom_list:    Canonical 132-symptom list from the model registry.
        model:           Optional fitted ``DecisionTreeClassifier`` for fallback ranking.
        max_questions:   Maximum number of tokens to return (default 8).

    Returns:
        Ordered list of symptom token strings ready to enqueue as follow-up
        questions.
    """
    asked_set: set[str] = set(already_asked)
    family_key = _match_family_key(primary_symptom)

    if family_key is not None:
        candidates = SYMPTOM_FAMILIES[family_key]
        queue = [s for s in candidates if s not in asked_set]
        logger.debug(
            "Symptom queue built from family",
            extra={
                "primary": primary_symptom,
                "family": family_key,
                "queue_length": len(queue),
            },
        )
        return queue[:max_questions]

    # Primary symptom not in any family — use model feature importances.
    if model is not None:
        logger.debug(
            "No family matched — falling back to feature importances",
            extra={"primary": primary_symptom},
        )
        return _fallback_from_importances(model, symptom_list, asked_set, max_questions)

    # Last resort: symptom_list order, excluding already asked.
    logger.debug(
        "No family and no model — using symptom_list order",
        extra={"primary": primary_symptom},
    )
    return [s for s in symptom_list if s not in asked_set][:max_questions]
