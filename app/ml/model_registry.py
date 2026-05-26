"""Singleton artifact registry — loaded exactly once per process lifetime.

Exposes the trained ``DecisionTreeClassifier``, the fitted ``LabelEncoder``,
the ordered 132-symptom list, and the full metadata dict.

The ``load()`` method is called during application startup (lifespan).
All other layers call the ``get_*`` helpers; they raise
``ModelNotInitialisedError`` if invoked before ``load()``.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from app.core.exceptions import ArtifactLoadError, ModelNotInitialisedError

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ("model.joblib", "label_encoder.joblib", "metadata.json")
_OPTIONAL_FILES = ("disease_info.json", "feature_importance.json")


class ModelRegistry:
    """Holds every ML artifact needed for inference.

    This class is intentionally not thread-locked because the model is loaded
    once before any request is served, and all subsequent access is read-only.
    """

    def __init__(self) -> None:
        self._model: DecisionTreeClassifier | None = None
        self._label_encoder: LabelEncoder | None = None
        self._symptom_list: list[str] | None = None
        self._metadata: dict[str, Any] | None = None
        self._disease_info: dict[str, Any] | None = None
        self._feature_importance: dict[str, float] | None = None
        self._initialised: bool = False

    def load(self, artifacts_dir: Path) -> None:
        """Load all artifacts from *artifacts_dir*.

        Args:
            artifacts_dir: Directory containing ``model.joblib``,
                ``label_encoder.joblib``, ``metadata.json``, and optionally
                ``disease_info.json``.

        Raises:
            ArtifactLoadError: If any required file is missing or corrupt.
        """
        for filename in _REQUIRED_FILES:
            if not (artifacts_dir / filename).exists():
                raise ArtifactLoadError(
                    f"Required artifact not found: {artifacts_dir / filename}"
                )

        try:
            self._model = joblib.load(artifacts_dir / "model.joblib")
            self._label_encoder = joblib.load(
                artifacts_dir / "label_encoder.joblib"
            )
            with open(artifacts_dir / "metadata.json", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            self._symptom_list = self._metadata["symptom_list"]
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to deserialise artifacts in {artifacts_dir}: {exc}"
            ) from exc

        # disease_info.json is optional; fall back to empty dict.
        disease_info_path = artifacts_dir / "disease_info.json"
        if disease_info_path.exists():
            try:
                with open(disease_info_path, encoding="utf-8") as fh:
                    self._disease_info = json.load(fh)
            except Exception as exc:
                logger.warning(
                    "Could not load disease_info.json — descriptions and "
                    "precautions will be empty.",
                    extra={"error": str(exc)},
                )
                self._disease_info = {}
        else:
            self._disease_info = {}

        # feature_importance.json is optional; derive from model if absent.
        fi_path = artifacts_dir / "feature_importance.json"
        if fi_path.exists():
            try:
                with open(fi_path, encoding="utf-8") as fh:
                    self._feature_importance = json.load(fh)
            except Exception as exc:
                logger.warning(
                    "Could not load feature_importance.json — will derive from model.",
                    extra={"error": str(exc)},
                )
                self._feature_importance = None
        else:
            self._feature_importance = None

        # Derive feature importance from model if not loaded from file.
        if self._feature_importance is None and self._model is not None:
            import numpy as np  # noqa: PLC0415
            importances = self._model.feature_importances_
            self._feature_importance = {
                sym: float(importances[i])
                for i, sym in enumerate(self._symptom_list or [])
            }
            logger.debug("Feature importances derived from model weights")

        self._initialised = True
        logger.info(
            "ModelRegistry initialised",
            extra={"artifacts_dir": str(artifacts_dir)},
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def _check(self) -> None:
        if not self._initialised:
            raise ModelNotInitialisedError(
                "ModelRegistry has not been loaded. "
                "Call model_registry.load() during application startup."
            )

    def get_model(self) -> DecisionTreeClassifier:
        """Return the loaded ``DecisionTreeClassifier``.

        Returns:
            The fitted classifier.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._model is not None  # noqa: S101  (guarded by _check)
        return self._model

    def get_label_encoder(self) -> LabelEncoder:
        """Return the fitted ``LabelEncoder`` for prognosis labels.

        Returns:
            The fitted encoder.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._label_encoder is not None  # noqa: S101
        return self._label_encoder

    def get_symptom_list(self) -> list[str]:
        """Return the ordered list of 132 canonical symptom names.

        Returns:
            Ordered symptom list used during training.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._symptom_list is not None  # noqa: S101
        return self._symptom_list

    def get_metadata(self) -> dict[str, Any]:
        """Return the full ``metadata.json`` dictionary.

        Returns:
            Dict with at least ``symptom_list``, ``class_names``,
            ``model_version``, and ``accuracy`` keys.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._metadata is not None  # noqa: S101
        return self._metadata

    def get_disease_info(self) -> dict[str, Any]:
        """Return the disease description/precaution lookup.

        Returns:
            Dict mapping disease name → ``{"description": str, "precautions": [str]}``.
            Empty dict if ``disease_info.json`` was not found.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._disease_info is not None  # noqa: S101
        return self._disease_info

    def get_feature_importance(self) -> dict[str, float]:
        """Return the symptom → Gini importance mapping.

        Returns:
            Dict mapping canonical symptom name → importance score in [0, 1].
            Derived from ``model.feature_importances_`` if
            ``feature_importance.json`` was not found on disk.

        Raises:
            ModelNotInitialisedError: If ``load()`` has not been called.
        """
        self._check()
        assert self._feature_importance is not None  # noqa: S101
        return self._feature_importance


# Module-level singleton — imported and called by lifespan, then read-only.
model_registry = ModelRegistry()
