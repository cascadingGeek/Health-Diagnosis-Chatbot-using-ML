"""Free-text symptom parser — deterministic fallback for the LLM pipeline.

Layer 1 (Claude) is the primary free-text parser.  This module provides an
offline, rule-based fallback that extracts symptom tokens from a user
sentence without making an API call.  Used when the LLM is unavailable or
as a pre-filter before calling Layer 1 to reduce input noise.

Strategy
--------
1. Split input into n-grams (1-, 2-, 3-word windows).
2. For each n-gram, try exact synonym lookup first (symptom_synonyms).
3. Fall back to rapidfuzz fuzzy matching against the full vocabulary.
4. Return deduplicated list of matched canonical symptom tokens.
"""

from __future__ import annotations

import logging
import re

from rapidfuzz import process as rfprocess

from app.core.symptom_synonyms import SYNONYM_MAP

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD: float = 75.0
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "i", "have", "been", "having", "feel", "feeling", "am", "is", "are",
        "the", "a", "an", "and", "or", "but", "with", "my", "me", "very",
        "really", "quite", "some", "since", "for", "of", "in", "on", "at",
        "to", "do", "not", "no", "yes", "also", "too", "just", "so", "about",
        "little", "bit", "kind", "this", "that", "it", "there", "here",
        "little", "slightly", "bad", "worse", "better", "much",
    }
)


def _normalise(text: str) -> str:
    """Lowercase and replace whitespace/hyphens with underscores."""
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate space-joined n-grams from a token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def parse_symptoms(
    text: str,
    symptom_vocabulary: list[str],
) -> list[str]:
    """Extract canonical symptom tokens from free-form user text.

    Args:
        text: Raw user input (any length).
        symptom_vocabulary: Ordered list of canonical symptom tokens.

    Returns:
        Deduplicated list of matched symptom tokens in order of appearance.
    """
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    content_tokens = [t for t in raw_tokens if t not in _STOP_WORDS]

    matched: list[str] = []
    seen: set[str] = set()

    normalised_vocab = [_normalise(s) for s in symptom_vocabulary]

    # Try 3-grams, then 2-grams, then 1-grams to prefer longer matches.
    for window_size in (3, 2, 1):
        for gram in _ngrams(content_tokens, window_size):
            # Synonym lookup first.
            synonym_match = SYNONYM_MAP.get(gram)
            if synonym_match and synonym_match not in seen:
                matched.append(synonym_match)
                seen.add(synonym_match)
                continue

            # Fuzzy match against the vocabulary.
            normalised_gram = _normalise(gram)
            result = rfprocess.extractOne(
                normalised_gram,
                normalised_vocab,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if result is not None:
                _, _, idx = result
                canonical = symptom_vocabulary[idx]
                if canonical not in seen:
                    matched.append(canonical)
                    seen.add(canonical)

    if matched:
        logger.debug(
            "NLP parser extracted %d symptoms from text: %s",
            len(matched),
            matched,
        )
    return matched
