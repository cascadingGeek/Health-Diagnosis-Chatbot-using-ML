"""Known wrong-prediction corrections for the Decision Tree.

The Kaggle synthetic dataset contains a small number of symptom patterns
where the trained Decision Tree consistently predicts an incorrect disease
due to overlapping features in the training data.  This module corrects
those known cases deterministically before the result reaches Layer 3.

Each override specifies:
  - wrong_disease: The incorrect prediction to intercept.
  - required_symptoms: All of these must be present in confirmed_symptoms.
  - excluded_symptoms: None of these may be present in confirmed_symptoms.
  - correct_disease: The replacement prediction.

Overrides are checked in list order; the first match wins.
If no override matches the input is passed through unchanged.

IMPORTANT: This file must be updated — not Layer 1 or Layer 3 — when a new
systematic wrong prediction is discovered.  The Decision Tree's output should
never be modified outside of this explicitly documented override list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Override:
    """Definition of a single disease prediction correction.

    Attributes:
        wrong_disease:     The Decision Tree prediction to intercept.
        required_symptoms: All must be present to trigger the override.
        excluded_symptoms: Any present here blocks the override.
        correct_disease:   Replacement disease name.
        reason:            Human-readable explanation for documentation.
    """

    wrong_disease: str
    required_symptoms: list[str]
    excluded_symptoms: list[str]
    correct_disease: str
    reason: str


# Documented overrides only.  Do not add speculative corrections.
OVERRIDES: list[Override] = [
    Override(
        wrong_disease="Common Cold",
        required_symptoms=["high_fever", "chills", "sweating", "headache", "nausea"],
        excluded_symptoms=["runny_nose", "continuous_sneezing", "throat_irritation"],
        correct_disease="Malaria",
        reason=(
            "High fever + chills + sweating + headache + nausea without cold-specific "
            "respiratory symptoms is a Malaria pattern, not a Common Cold pattern."
        ),
    ),
    Override(
        wrong_disease="Fungal infection",
        required_symptoms=["yellowish_skin", "dark_urine", "abdominal_pain", "vomiting"],
        excluded_symptoms=["itching", "skin_rash", "nodal_skin_eruptions"],
        correct_disease="Hepatitis A",
        reason=(
            "Jaundice + dark urine + abdominal pain + vomiting without skin symptoms "
            "maps to Hepatitis A, not Fungal infection."
        ),
    ),
]


def apply_overrides(
    predicted_disease: str,
    confirmed_symptoms: list[str],
) -> str:
    """Apply documented prediction corrections to a Decision Tree result.

    Args:
        predicted_disease: Raw prediction from the Decision Tree.
        confirmed_symptoms: Symptom tokens confirmed during the session.

    Returns:
        The corrected disease name, or *predicted_disease* unchanged if no
        override matches.
    """
    confirmed_set = set(confirmed_symptoms)

    for override in OVERRIDES:
        if override.wrong_disease != predicted_disease:
            continue

        required_met = all(s in confirmed_set for s in override.required_symptoms)
        excluded_absent = not any(s in confirmed_set for s in override.excluded_symptoms)

        if required_met and excluded_absent:
            logger.info(
                "Disease override applied: %s → %s | reason: %s",
                predicted_disease,
                override.correct_disease,
                override.reason,
            )
            return override.correct_disease

    return predicted_disease
