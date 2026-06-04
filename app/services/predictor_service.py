"""Layer 2 of the hybrid diagnostic pipeline.

Thin wrapper around the existing diagnosis_service that exposes the
Decision Tree prediction as a plain dict for the LLM orchestrator.

The Decision Tree makes every diagnosis.  This layer only runs the model
and returns the raw result — it does not validate, explain, or modify it.
"""

import logging

import numpy as np

from app.ml.feature_builder import build_vector
from app.ml.model_registry import model_registry
from app.ml.predictor import predict

logger = logging.getLogger(__name__)


def predict_disease(confirmed_symptoms: list[str]) -> dict:
    """Run the Decision Tree on confirmed symptoms and return raw prediction.

    Bypasses the confidence threshold check so that Layer 3 (Claude) can
    evaluate plausibility regardless of the raw score.

    Args:
        confirmed_symptoms: Symptom tokens confirmed during dialogue.

    Returns:
        Dict with keys:
          - disease (str | None): Predicted disease name, or None if all
            symptom tokens were out-of-vocabulary.
          - confidence (float): Raw predict_proba max value in [0, 1],
            or 0.0 if disease is None.

    Notes:
        Unknown symptom tokens are silently skipped by ``build_vector``.
        When no valid symptoms remain the returned dict has
        ``disease=None, confidence=0.0``, which the orchestrator treats as
        a fallback trigger.
    """
    symptom_list = model_registry.get_symptom_list()
    vector = build_vector(confirmed_symptoms, symptom_list)

    # If every symptom was skipped (all out-of-vocabulary), return a sentinel
    # that the orchestrator recognises as a fallback trigger.
    if not np.any(vector):
        logger.warning(
            "All %d symptom(s) were out-of-vocabulary — returning null prediction",
            len(confirmed_symptoms),
        )
        return {"disease": None, "confidence": 0.0}

    disease, confidence = predict(vector, model_registry)

    logger.info(
        "Layer 2 Decision Tree prediction: disease=%s raw_confidence=%.3f "
        "n_symptoms=%d",
        disease,
        confidence,
        len(confirmed_symptoms),
    )
    return {"disease": disease, "confidence": confidence}
