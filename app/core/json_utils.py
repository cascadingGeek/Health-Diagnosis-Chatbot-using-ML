"""
Shared JSON extraction utility for the diagnostic pipeline.

Claude API responses — especially when web search tools are used —
frequently contain prose before and after the JSON block.

Examples of what the raw response looks like::

    "Based on my research...\\n\\n```json\\n{...}\\n```"
    "Here is the assessment:\\n```json\\n{...}\\n```"
    "{...}"                          # clean JSON, no prose
    "```json\\n{...}\\n```"          # fence with no prose prefix

This utility handles all four cases reliably.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json_from_response(raw: str) -> dict:
    """Extract and parse a JSON object from a Claude API response string.

    Tries four strategies in order of strictness:
    1. Direct parse — fast path for clean JSON responses.
    2. Markdown fence extraction — handles ```json ... ``` anywhere in the string.
    3. First-brace to last-brace extraction — handles prose before/after JSON.
    4. Brute-force brace-block scan — last resort for unusual formatting.

    Args:
        raw: The raw text content from a Claude API response block.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object can be found in the string.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response string")

    text = raw.strip()

    # Strategy 1: Direct parse — works when response is clean JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fence anywhere in the string.
    # Matches ```json ... ``` or ``` ... ``` with a JSON object inside.
    fence_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        re.DOTALL,
    )
    match = fence_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.debug("Fence extraction found block but parse failed: %s", exc)

    # Strategy 3: Find the outermost { ... } span in the whole string.
    # Handles prose-before and prose-after, including web search preambles.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.debug("Brace-span extraction failed: %s", exc)

    # Strategy 4: Find all top-level {...} blocks and try each one.
    # Useful when there are multiple JSON fragments and only one is valid.
    brace_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    for candidate in reversed(brace_pattern.findall(text)):  # largest/last first
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"No valid JSON object found in response. "
        f"First 200 chars: {text[:200]!r}"
    )
