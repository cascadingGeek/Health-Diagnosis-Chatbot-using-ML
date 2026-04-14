"""Unit tests for app/ml/feature_builder.py."""

import numpy as np
import pytest

from app.core.exceptions import FeatureContractError
from app.ml.feature_builder import _normalise, build_vector

# ── Fixtures ──────────────────────────────────────────────────────────────────

SYMPTOM_LIST = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
]


# ── _normalise ─────────────────────────────────────────────────────────────────


class TestNormalise:
    def test_lower_cases(self):
        assert _normalise("FEVER") == "fever"

    def test_replaces_spaces_with_underscore(self):
        assert _normalise("joint pain") == "joint_pain"

    def test_replaces_hyphens(self):
        assert _normalise("skin-rash") == "skin_rash"

    def test_strips_whitespace(self):
        assert _normalise("  chills  ") == "chills"

    def test_mixed(self):
        assert _normalise("  Joint Pain  ") == "joint_pain"


# ── build_vector ──────────────────────────────────────────────────────────────


class TestBuildVector:
    def test_shape_is_1_by_n(self):
        vec = build_vector(["itching"], SYMPTOM_LIST)
        assert vec.shape == (1, len(SYMPTOM_LIST))

    def test_dtype_is_float64(self):
        vec = build_vector(["itching"], SYMPTOM_LIST)
        assert vec.dtype == np.float64

    def test_single_symptom_sets_correct_position(self):
        vec = build_vector(["itching"], SYMPTOM_LIST)
        assert vec[0, 0] == 1.0
        assert vec[0, 1:].sum() == 0.0

    def test_multiple_symptoms(self):
        vec = build_vector(["itching", "joint_pain"], SYMPTOM_LIST)
        assert vec[0, 0] == 1.0
        assert vec[0, 6] == 1.0
        assert vec[0].sum() == 2.0

    def test_exact_match_with_spaces(self):
        """'joint pain' should fuzzy-match 'joint_pain'."""
        vec = build_vector(["joint pain"], SYMPTOM_LIST)
        assert vec[0, 6] == 1.0

    def test_exact_match_mixed_case(self):
        vec = build_vector(["ITCHING"], SYMPTOM_LIST)
        assert vec[0, 0] == 1.0

    def test_unknown_symptom_raises(self):
        with pytest.raises(FeatureContractError) as exc_info:
            build_vector(["completely_unknown_xyz"], SYMPTOM_LIST)
        assert "completely_unknown_xyz" in str(exc_info.value)

    def test_empty_symptom_list_gives_zero_vector(self):
        vec = build_vector([], SYMPTOM_LIST)
        assert vec.sum() == 0.0
        assert vec.shape == (1, len(SYMPTOM_LIST))
