"""Anchor symptom validation for high-stakes disease predictions.

Defines the minimum set of anchor symptoms that must be present for a
high-stakes disease prediction to be considered plausible.  Used as a
deterministic safety check on top of Layer 3's (Claude) plausibility
validation.

If the Decision Tree predicts a disease in ANCHOR_RULES but none of its
anchor symptoms appear in the confirmed list, the guard raises
``AnchorViolationError`` and the response is forced to "inconclusive".

This is a defence-in-depth measure.  Layer 3 performs the primary plausibility
check; this guard catches the same edge cases offline without an API call.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AnchorViolationError(Exception):
    """Raised when a high-stakes prediction has no anchor symptoms."""


# Maps predicted disease name → minimum required anchor symptoms.
# At least ONE symptom from the list must appear in confirmed_symptoms.
ANCHOR_RULES: dict[str, list[str]] = {
    "Heart attack": ["chest_pain", "fast_heart_rate", "breathlessness", "vomiting"],
    "AIDS": [
        "muscle_wasting",
        "patches_in_throat",
        "swelled_lymph_nodes",
        "extra_marital_contacts",
    ],
    "Tuberculosis": ["cough", "blood_in_sputum", "rusty_sputum"],
    "Stroke": [
        "weakness_of_one_body_side",
        "slurred_speech",
        "loss_of_balance",
        "altered_sensorium",
    ],
    "Hepatitis B": ["yellowish_skin", "vomiting", "dark_urine", "abdominal_pain"],
    "Hepatitis C": ["yellowish_skin", "vomiting", "dark_urine", "fatigue"],
    "Hepatitis D": ["yellowish_skin", "vomiting", "dark_urine", "abdominal_pain"],
    "Hepatitis E": ["yellowish_skin", "vomiting", "dark_urine", "abdominal_pain"],
    "Cirrhosis": ["yellowish_skin", "swelling_of_stomach", "fluid_overload", "fatigue"],
}


def check_anchors(predicted_disease: str, confirmed_symptoms: list[str]) -> None:
    """Raise AnchorViolationError if anchor constraints are violated.

    Only diseases listed in ANCHOR_RULES are checked.  All other predictions
    pass through without validation.

    Args:
        predicted_disease: Disease name returned by the Decision Tree.
        confirmed_symptoms: List of symptom tokens confirmed during dialogue.

    Raises:
        AnchorViolationError: When none of the required anchor symptoms are
            present in *confirmed_symptoms* for a high-stakes disease.
    """
    anchors = ANCHOR_RULES.get(predicted_disease)
    if anchors is None:
        return  # Not a high-stakes disease — no check needed.

    confirmed_set = set(confirmed_symptoms)
    if not confirmed_set.intersection(anchors):
        logger.warning(
            "Anchor violation: predicted=%s but none of %s in confirmed=%s",
            predicted_disease,
            anchors,
            confirmed_symptoms,
        )
        raise AnchorViolationError(
            f"Prediction '{predicted_disease}' has no matching anchor symptoms. "
            f"Required one of: {anchors}"
        )
