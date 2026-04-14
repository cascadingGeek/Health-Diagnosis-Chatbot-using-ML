"""Business logic layer for symptom diagnosis.

Wraps the ML predictor with confidence thresholding and result formatting.
Has no knowledge of HTTP, FastAPI, or the database.
"""

import logging

from app.core.config import settings
from app.core.exceptions import FeatureContractError, InconclusiveDiagnosisError
from app.ml.feature_builder import build_vector
from app.ml.model_registry import ModelRegistry
from app.ml.predictor import build_diagnosis_result, predict
from app.schemas.diagnosis import DiagnosisResult

logger = logging.getLogger(__name__)


def diagnose(
    confirmed_symptoms: list[str],
    registry: ModelRegistry,
    *,
    confidence_threshold: float | None = None,
) -> DiagnosisResult:
    """Run end-to-end diagnosis for a confirmed symptom list.

    Args:
        confirmed_symptoms:   Symptom strings collected during dialogue.
        registry:             Initialised ``ModelRegistry``.
        confidence_threshold: Override the global threshold for testing.
            Defaults to ``settings.confidence_threshold``.

    Returns:
        A populated ``DiagnosisResult``.

    Raises:
        FeatureContractError:      If a symptom cannot be matched.
        InconclusiveDiagnosisError: If confidence is below the threshold.
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.confidence_threshold
    )
    symptom_list = registry.get_symptom_list()

    # May raise FeatureContractError — let it propagate to the controller.
    vector = build_vector(confirmed_symptoms, symptom_list)

    disease, confidence = predict(vector, registry)

    logger.info(
        "Diagnosis result",
        extra={
            "disease": disease,
            "confidence": confidence,
            "threshold": threshold,
            "n_symptoms": len(confirmed_symptoms),
        },
    )

    if confidence < threshold:
        raise InconclusiveDiagnosisError(
            f"Confidence {confidence:.2%} is below threshold {threshold:.2%}"
        )

    return build_diagnosis_result(disease, confidence, registry)
