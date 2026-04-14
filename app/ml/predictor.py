"""Pure inference wrapper around the loaded DecisionTreeClassifier.

This module has zero FastAPI, zero database, and zero HTTP knowledge.
It receives a pre-built feature vector and returns a structured result.
"""

import logging

import numpy as np

from app.ml.model_registry import ModelRegistry
from app.schemas.diagnosis import DiagnosisResult

logger = logging.getLogger(__name__)


def predict(
    symptom_vector: np.ndarray,
    registry: ModelRegistry,
) -> tuple[str, float]:
    """Run inference and return the predicted disease and confidence.

    Args:
        symptom_vector: Binary array of shape ``(1, n_features)``.
        registry:       Initialised ``ModelRegistry`` instance.

    Returns:
        A tuple of ``(disease_name, confidence)`` where confidence is the
        probability of the predicted class in ``[0, 1]``.
    """
    model = registry.get_model()
    label_encoder = registry.get_label_encoder()

    proba = model.predict_proba(symptom_vector)  # shape (1, n_classes)
    class_idx = int(np.argmax(proba[0]))
    confidence = float(proba[0, class_idx])
    disease = label_encoder.inverse_transform([class_idx])[0]

    logger.debug(
        "Prediction complete",
        extra={"disease": disease, "confidence": confidence},
    )
    return str(disease), confidence


def build_diagnosis_result(
    disease: str,
    confidence: float,
    registry: ModelRegistry,
) -> DiagnosisResult:
    """Assemble a ``DiagnosisResult`` by enriching the raw prediction.

    Looks up description and precautions from the disease_info registry.
    Falls back to empty strings / lists if the disease is not in the lookup.

    Args:
        disease:    Predicted disease name.
        confidence: Model confidence score.
        registry:   Initialised ``ModelRegistry`` instance.

    Returns:
        A fully populated ``DiagnosisResult``.
    """
    disease_info = registry.get_disease_info()
    info = disease_info.get(disease, {})
    return DiagnosisResult(
        disease=disease,
        confidence=confidence,
        description=info.get("description", ""),
        precautions=info.get("precautions", []),
    )
