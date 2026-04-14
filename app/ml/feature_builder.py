"""Convert a list of confirmed symptom strings into a binary feature vector.

The symptom_list in metadata.json defines the canonical order of all 132
features used during training.  This module maps user-facing symptom strings
to their position in that list, raising ``FeatureContractError`` for any
unrecognised name.

Fuzzy matching via ``rapidfuzz`` is used to normalise minor spelling variations
before the strict look-up, so user input like ``"joint pain"`` maps correctly
to ``"joint_pain"`` in the training vocabulary.
"""

import logging
import re

import numpy as np
from rapidfuzz import process as rfprocess

from app.core.exceptions import FeatureContractError

logger = logging.getLogger(__name__)

# Minimum similarity score (0–100) required for a fuzzy match to be accepted.
_FUZZY_THRESHOLD: float = 80.0


def _normalise(name: str) -> str:
    """Lower-case and replace whitespace / hyphens with underscores.

    Args:
        name: Raw symptom string from the user or the candidate list.

    Returns:
        Normalised string suitable for comparison.
    """
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _resolve_symptom(raw: str, symptom_list: list[str]) -> str:
    """Map a raw symptom string to its canonical name in *symptom_list*.

    First attempts an exact match after normalisation.  Falls back to fuzzy
    matching via ``rapidfuzz.process.extractOne``.

    Args:
        raw:          User-supplied or dialogue-extracted symptom name.
        symptom_list: Ordered list of canonical symptom names from metadata.

    Returns:
        The matched canonical symptom name.

    Raises:
        FeatureContractError: If no match with sufficient confidence is found.
    """
    normalised = _normalise(raw)
    normalised_list = [_normalise(s) for s in symptom_list]

    # Exact match first (fast path).
    if normalised in normalised_list:
        idx = normalised_list.index(normalised)
        return symptom_list[idx]

    # Fuzzy fallback.
    result = rfprocess.extractOne(
        normalised,
        normalised_list,
        score_cutoff=_FUZZY_THRESHOLD,
    )
    if result is None:
        raise FeatureContractError(raw)

    matched_normalised, score, idx = result
    canonical = symptom_list[idx]
    logger.debug(
        "Fuzzy-matched symptom",
        extra={"raw": raw, "matched": canonical, "score": score},
    )
    return canonical


def build_vector(
    collected_symptoms: list[str],
    symptom_list: list[str],
) -> np.ndarray:
    """Build a binary feature vector from a list of confirmed symptoms.

    Args:
        collected_symptoms: Symptom strings confirmed during dialogue.
        symptom_list:       Ordered canonical symptom list (length 132).

    Returns:
        ``np.ndarray`` of shape ``(1, len(symptom_list))`` with ``1.0`` at
        each position corresponding to a confirmed symptom.

    Raises:
        FeatureContractError: If any symptom cannot be matched.
    """
    vector = np.zeros((1, len(symptom_list)), dtype=np.float64)

    for raw in collected_symptoms:
        canonical = _resolve_symptom(raw, symptom_list)
        idx = symptom_list.index(canonical)
        vector[0, idx] = 1.0

    return vector
