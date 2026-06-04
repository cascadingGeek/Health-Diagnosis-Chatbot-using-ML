"""
Layer 3 of the hybrid diagnostic pipeline.

Responsibility: Clinical validation and response generation only.
- Receives the Decision Tree's prediction (disease + confidence + decision path)
- Validates that the prediction is clinically plausible given the confirmed symptoms
- Generates a human-readable, empathetic diagnosis explanation
- Handles out-of-dataset diseases using web search when available
- Always appends the medical disclaimer

This layer does NOT change the predicted disease class.
It can flag a prediction as implausible and return an inconclusive response instead.
The Decision Tree prediction is treated as ground truth unless clearly incoherent.
"""

import logging

from anthropic import Anthropic

from app.core.config import settings
from app.core.json_utils import extract_json_from_response

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)

MEDICAL_DISCLAIMER = (
    "⚠️ Important Disclaimer: This is a preliminary AI-assisted assessment only. "
    "It is not a substitute for professional medical diagnosis or treatment. "
    "The results are based on the symptoms you described and the system may not be "
    "fully accurate. Please consult a qualified doctor or visit a healthcare facility "
    "before making any health decisions, especially if your symptoms are severe or worsening."
)

LAYER3_SYSTEM_PROMPT = """You are a medical response generator working as the final layer
of a diagnostic pipeline. A machine learning Decision Tree classifier has already made
a diagnosis prediction. Your job is to validate and explain it — not to override it.

YOUR RESPONSIBILITIES:
1. Check if the predicted disease is clinically plausible given the confirmed symptoms
2. If plausible: generate a clear, empathetic, plain-English explanation of the diagnosis
3. If implausible: return an inconclusive response directing the user to see a doctor
4. Provide 3-5 practical precautions or next steps
5. Indicate urgency level: mild / moderate / urgent
6. Never suggest a different disease than what the model predicted
7. Never remove or modify the disclaimer field

CLINICAL IMPLAUSIBILITY RULES - return inconclusive if:
- Predicted disease is "Heart Attack" but chest_pain, fast_heart_rate, and vomiting
  are all absent from confirmed symptoms
- Predicted disease is "AIDS" but muscle_wasting, patches_in_throat,
  swollen_lymph_nodes, and extra_marital_contacts are all absent
- Predicted disease is "Tuberculosis" but neither cough nor blood_in_sputum
  is in confirmed symptoms
- Predicted disease is "Stroke" but weakness_of_one_body_side, slurred_speech,
  and loss_of_balance are all absent
- Confidence score is below 0.60

TONE RULES:
- Tier 1 (mild - cold, cough, minor infections): reassuring, calm, brief
- Tier 2 (moderate - malaria, typhoid, UTI): informative, clear, action-oriented
- Tier 3 (serious - heart conditions, stroke, organ failure): direct, urgent,
  emphasise immediate professional care

RESPONSE FORMAT: Valid JSON only. No prose outside the JSON.
{{
  "is_plausible": true,
  "display_disease": "Disease Name",
  "display_confidence": 0.83,
  "urgency": "mild | moderate | urgent",
  "explanation": "Plain English explanation of what this condition is and why the symptoms match",
  "precautions": [
    "Specific actionable step 1",
    "Specific actionable step 2",
    "Specific actionable step 3"
  ],
  "when_to_see_doctor": "Specific guidance on when to seek care",
  "disclaimer": "ALWAYS copy this exactly: {disclaimer}"
}}

If is_plausible is false:
{{
  "is_plausible": false,
  "display_disease": null,
  "display_confidence": null,
  "urgency": "moderate",
  "explanation": "Your symptoms do not clearly match a single condition in our system.",
  "precautions": [
    "Please visit the nearest clinic or hospital for a proper evaluation",
    "Describe all your symptoms to the doctor as you described them to this system",
    "Do not self-medicate based on this assessment"
  ],
  "when_to_see_doctor": "As soon as possible",
  "disclaimer": "{disclaimer}"
}}
"""


def validate_and_explain_diagnosis(
    predicted_disease: str,
    raw_confidence: float,
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    conversation_summary: str,
    session_id: str = "",
) -> dict:
    """Layer 3 entry point. Validates and explains a Decision Tree prediction.

    Args:
        predicted_disease: Disease name from the Decision Tree.
        raw_confidence: Raw predict_proba max value (0.0-1.0).
        confirmed_symptoms: Symptoms the user confirmed.
        denied_symptoms: Symptoms the user denied.
        conversation_summary: Brief summary of the full dialogue.
        session_id: Optional session identifier for log correlation.

    Returns:
        Parsed response dict ready to send to the frontend.
    """
    display_confidence = dampen_confidence(raw_confidence)

    user_message = (
        f"Decision Tree prediction: {predicted_disease}\n"
        f"Raw confidence: {raw_confidence:.3f}\n"
        f"Display confidence: {display_confidence:.3f}\n"
        f"Confirmed symptoms: {confirmed_symptoms}\n"
        f"Denied symptoms: {denied_symptoms}\n"
        f"Conversation summary: {conversation_summary}\n\n"
        f"Validate this prediction and generate the response JSON."
    )

    # web_search_20250305 allows Layer 3 to look up out-of-dataset diseases.
    # The tool dict is passed as-is to the API; older SDK versions forward it unchanged.
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
        }
    ]

    raw = ""
    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=LAYER3_SYSTEM_PROMPT.format(disclaimer=MEDICAL_DISCLAIMER),
            messages=[{"role": "user", "content": user_message}],
            tools=tools,
        )

        logger.info(
            "Claude API usage | model=%s input_tokens=%d output_tokens=%d "
            "session=%s layer=layer3",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            session_id,
        )

        # Response may include tool_use blocks; take the last text block.
        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            logger.warning(
                "Layer 3 returned no text block — returning safe fallback | session=%s",
                session_id,
            )
            return _inconclusive_fallback()

        raw = text_blocks[-1].text.strip()
        try:
            parsed = extract_json_from_response(raw)
        except ValueError as exc:
            logger.error(
                "Layer 3 JSON extraction failed | error=%s | raw_preview=%s | session=%s",
                exc, raw[:300], session_id,
            )
            return _inconclusive_fallback()

        # Safety: disclaimer must always be present.
        if not parsed.get("disclaimer"):
            parsed["disclaimer"] = MEDICAL_DISCLAIMER

        # Safety: display_confidence must never exceed 0.87 (new high-tier ceiling).
        if parsed.get("display_confidence") and parsed["display_confidence"] > 0.87:
            parsed["display_confidence"] = 0.87

        logger.info(
            "Layer 3 result: plausible=%s disease=%s confidence=%s urgency=%s | session=%s",
            parsed.get("is_plausible"),
            parsed.get("display_disease"),
            parsed.get("display_confidence"),
            parsed.get("urgency"),
            session_id,
        )
        return parsed
    except Exception as exc:
        logger.error("Layer 3 API error: %s | session=%s", exc, session_id)
        # If web_search tool is unsupported by this SDK version, retry without it.
        if "tool" in str(exc).lower() or "web_search" in str(exc).lower():
            logger.info(
                "Retrying Layer 3 without web_search tool | session=%s", session_id
            )
            return _validate_without_web_search(
                user_message, session_id, display_confidence
            )
        return _inconclusive_fallback()


def _validate_without_web_search(
    user_message: str,
    session_id: str,
    display_confidence: float,
) -> dict:
    """Fallback Layer 3 call that omits the web_search tool.

    Used when the Anthropic SDK version does not support the web_search_20250305
    tool type.

    Args:
        user_message: The pre-formatted user message for Layer 3.
        session_id: Session identifier for log correlation.
        display_confidence: Already-dampened confidence value.

    Returns:
        Parsed response dict or inconclusive fallback.
    """
    raw = ""
    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=LAYER3_SYSTEM_PROMPT.format(disclaimer=MEDICAL_DISCLAIMER),
            messages=[{"role": "user", "content": user_message}],
        )

        logger.info(
            "Claude API usage (no-web-search retry) | model=%s input_tokens=%d "
            "output_tokens=%d session=%s layer=layer3",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            session_id,
        )

        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            return _inconclusive_fallback()

        raw = text_blocks[-1].text.strip()
        try:
            parsed = extract_json_from_response(raw)
        except ValueError as exc:
            logger.error(
                "Layer 3 (no-web-search) JSON extraction failed | error=%s | "
                "raw_preview=%s | session=%s",
                exc, raw[:300], session_id,
            )
            return _inconclusive_fallback()
        if not parsed.get("disclaimer"):
            parsed["disclaimer"] = MEDICAL_DISCLAIMER
        if parsed.get("display_confidence") and parsed["display_confidence"] > 0.87:
            parsed["display_confidence"] = 0.87
        return parsed

    except Exception as exc:
        logger.error(
            "Layer 3 no-web-search fallback also failed: %s | session=%s",
            exc,
            session_id,
        )
        return _inconclusive_fallback()


def dampen_confidence(raw: float) -> float:
    """Transform raw Decision Tree probability to display confidence.

    Decision Trees trained on the Kaggle synthetic dataset memorise training
    patterns and return raw probabilities of 1.0 far too often.  This
    transformation makes the displayed score more honest in two ways:

    - High tier (raw 0.90–1.0): linearly mapped to 0.75–0.87, then a small
      random jitter (±0–0.02) is added.  The jitter prevents every high-
      confidence prediction from displaying the same round number, which
      would look artificial to users.  The result is always kept within
      [0.75, 0.87] after jitter.
    - Mid tier (raw 0.60–0.89): scaled by 0.92 (light dampening only).
    - Low tier (raw < 0.60): returned as-is.

    Does NOT affect the predicted class — only the displayed score.

    Args:
        raw: Raw predict_proba value from the Decision Tree (0.0–1.0).

    Returns:
        Dampened display confidence in [0.0, 0.87].
    """
    import random

    if raw >= 0.90:
        # Linear map: 0.90 → 0.75, 1.00 → 0.87
        scaled = 0.75 + (raw - 0.90) * (0.12 / 0.10)
        # Jitter: uniform ±0.02, seeded by raw so repeated calls with the
        # same input still vary (random, not deterministic).
        jitter = random.uniform(-0.02, 0.02)
        dampened = round(min(max(scaled + jitter, 0.75), 0.87), 2)
        return dampened
    elif raw >= 0.60:
        return round(raw * 0.92, 2)
    return round(raw, 2)


def _inconclusive_fallback() -> dict:
    """Safe fallback response when Layer 3 fails or returns implausible result."""
    return {
        "is_plausible": False,
        "display_disease": None,
        "display_confidence": None,
        "urgency": "moderate",
        "explanation": (
            "We were unable to determine a clear diagnosis from the symptoms described. "
            "This may be because your symptoms match multiple conditions or fall outside "
            "the conditions this system was trained on."
        ),
        "precautions": [
            "Visit the nearest clinic or hospital for a proper evaluation",
            "Describe all your symptoms to the doctor as you described them here",
            "Do not self-medicate based on this assessment",
        ],
        "when_to_see_doctor": "As soon as possible",
        "disclaimer": MEDICAL_DISCLAIMER,
    }
