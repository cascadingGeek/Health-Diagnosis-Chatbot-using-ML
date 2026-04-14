"""Unit tests for app/ml/predictor.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.ml.predictor import build_diagnosis_result, predict
from app.schemas.diagnosis import DiagnosisResult


def _make_registry(
    n_classes: int = 3,
    predicted_class_idx: int = 1,
    confidence: float = 0.85,
    class_names: list[str] | None = None,
    disease_info: dict | None = None,
) -> MagicMock:
    """Build a mock ModelRegistry for predictor tests."""
    if class_names is None:
        class_names = [f"Disease_{i}" for i in range(n_classes)]
    if disease_info is None:
        disease_info = {
            class_names[predicted_class_idx]: {
                "description": "A test disease.",
                "precautions": ["rest", "hydrate"],
            }
        }

    proba = np.zeros((1, n_classes))
    proba[0, predicted_class_idx] = confidence

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = proba

    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = [class_names[predicted_class_idx]]

    registry = MagicMock()
    registry.get_model.return_value = mock_model
    registry.get_label_encoder.return_value = mock_le
    registry.get_disease_info.return_value = disease_info
    return registry


class TestPredict:
    def test_returns_disease_and_confidence(self):
        registry = _make_registry(
            n_classes=3,
            predicted_class_idx=1,
            confidence=0.9,
            class_names=["Flu", "Cold", "Fever"],
        )
        vector = np.zeros((1, 132))
        disease, conf = predict(vector, registry)
        assert disease == "Cold"
        assert pytest.approx(conf, abs=1e-6) == 0.9

    def test_calls_predict_proba_with_vector(self):
        registry = _make_registry()
        vector = np.ones((1, 10))
        predict(vector, registry)
        registry.get_model().predict_proba.assert_called_once_with(vector)

    def test_selects_argmax_class(self):
        n = 5
        proba = np.array([[0.05, 0.05, 0.7, 0.1, 0.1]])
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = proba
        mock_le = MagicMock()
        mock_le.inverse_transform.return_value = ["Disease_2"]
        registry = MagicMock()
        registry.get_model.return_value = mock_model
        registry.get_label_encoder.return_value = mock_le
        disease, conf = predict(np.zeros((1, n)), registry)
        assert disease == "Disease_2"
        assert pytest.approx(conf) == 0.7


class TestBuildDiagnosisResult:
    def test_populates_all_fields(self):
        registry = _make_registry(
            n_classes=2,
            predicted_class_idx=0,
            class_names=["Malaria", "Dengue"],
            disease_info={
                "Malaria": {
                    "description": "Mosquito-borne disease.",
                    "precautions": ["use nets", "take medication"],
                }
            },
        )
        result = build_diagnosis_result("Malaria", 0.92, registry)
        assert isinstance(result, DiagnosisResult)
        assert result.disease == "Malaria"
        assert pytest.approx(result.confidence) == 0.92
        assert result.description == "Mosquito-borne disease."
        assert result.precautions == ["use nets", "take medication"]

    def test_unknown_disease_uses_empty_defaults(self):
        registry = MagicMock()
        registry.get_disease_info.return_value = {}
        result = build_diagnosis_result("UnknownDisease", 0.75, registry)
        assert result.description == ""
        assert result.precautions == []
