"""
Layer 1 of the hybrid diagnostic pipeline.

Responsibility: Natural language understanding only.
- Accepts free text from the user
- Extracts confirmed and denied symptoms from conversation history
- Generates the next intelligent follow-up question
- Maps natural language to dataset-compatible symptom tokens

This layer does NOT make diagnostic predictions.
That is exclusively the responsibility of the Decision Tree (Layer 2).
"""

import json
import logging

from anthropic import Anthropic

from app.core.config import settings

logger = logging.getLogger(__name__)

# Client is created once at module load; the API key comes from settings
# (which reads .env) — never hardcoded.
_client = Anthropic(api_key=settings.anthropic_api_key)

# The full 132-symptom vocabulary from the Kaggle dataset.
# This list matches the canonical feature order used during model training.
DATASET_SYMPTOMS: list[str] = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps",
    "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain",
    "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell",
    "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "toxic_look_typhos",
    "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic_patches", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_stool", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption",
    "fluid_overload_1", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples",
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting",
    "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze",
]

_LAYER1_SYSTEM_PROMPT = """You are a medical symptom extraction assistant working as part of a diagnostic pipeline.

Your ONLY job is to:
1. Understand what symptoms the user is describing in natural language
2. Extract confirmed and denied symptoms from the conversation
3. Generate ONE clear, natural follow-up question when more information is needed
4. Map symptoms to tokens from the provided vocabulary list

You do NOT make diagnoses. You do NOT suggest diseases. You do NOT give medical advice.
Diagnosis is handled by a separate machine learning model downstream.

SYMPTOM VOCABULARY: You must map every symptom to the closest matching token from this list:
{symptom_vocabulary}

RESPONSE FORMAT: Always respond with valid JSON only. No prose. No markdown. No explanation outside the JSON.

For symptom extraction, use this format:
{{
  "action": "extract",
  "confirmed_symptoms": ["symptom_token_1", "symptom_token_2"],
  "denied_symptoms": ["symptom_token_3"],
  "unmapped_complaints": ["any complaint you could not map to the vocabulary"],
  "needs_more_info": true
}}

For follow-up questions, use this format:
{{
  "action": "ask",
  "question": "Your single natural follow-up question here",
  "target_symptom": "the_symptom_token_this_question_probes",
  "reasoning": "brief clinical reason why this question is next"
}}

For when enough symptoms are collected (minimum 3 confirmed symptoms), use:
{{
  "action": "ready",
  "confirmed_symptoms": ["symptom_1", "symptom_2", "symptom_3"],
  "denied_symptoms": ["symptom_5"],
  "summary": "Brief plain English summary of what the user reported"
}}

RULES:
- Never ask about a symptom already confirmed or denied in conversation history
- Never ask more than one question at a time
- Map colloquial terms to dataset tokens (e.g. "tummy ache" -> "stomach_pain", "tired" -> "fatigue")
- If a symptom has no close match in the vocabulary, add it to unmapped_complaints
- Minimum 3 confirmed symptoms before action: ready
- Maximum 8 follow-up questions total per session
"""


def extract_symptoms_from_text(
    user_message: str,
    conversation_history: list[dict],
    confirmed_so_far: list[str],
    denied_so_far: list[str],
    questions_asked: int,
    session_id: str = "",
) -> dict:
    """Layer 1 entry point. Processes a single user message in context.

    Args:
        user_message: The raw text from the user.
        conversation_history: Full list of {role, content} turns so far.
        confirmed_so_far: Symptoms confirmed in this session.
        denied_so_far: Symptoms denied in this session.
        questions_asked: Number of follow-up questions already asked.
        session_id: Optional session identifier for log correlation.

    Returns:
        Parsed dict with action, symptoms, and next question if applicable.
    """
    system = _LAYER1_SYSTEM_PROMPT.format(
        symptom_vocabulary=", ".join(DATASET_SYMPTOMS)
    )

    state_context = {
        "role": "user",
        "content": (
            f"Current session state:\n"
            f"- Confirmed symptoms so far: {confirmed_so_far}\n"
            f"- Denied symptoms so far: {denied_so_far}\n"
            f"- Follow-up questions asked: {questions_asked}/8\n"
            f"- Force ready if questions_asked >= 7\n\n"
            f'User just said: "{user_message}"\n\n'
            f"What is your action?"
        ),
    }

    messages = list(conversation_history) + [state_context]

    raw = ""
    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=512,
            system=system,
            messages=messages,
        )

        logger.info(
            "Claude API usage | model=%s input_tokens=%d output_tokens=%d "
            "session=%s layer=layer1",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            session_id,
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        logger.info("Layer 1 action: %s | session=%s", parsed.get("action"), session_id)
        return parsed

    except json.JSONDecodeError as exc:
        logger.error(
            "Layer 1 JSON parse error: %s | raw=%s | session=%s", exc, raw, session_id
        )
        return {
            "action": "ask",
            "question": (
                "Could you describe your main symptom again? "
                "I want to make sure I understand correctly."
            ),
            "target_symptom": None,
            "reasoning": "JSON parse fallback",
        }
    except Exception as exc:
        logger.error("Layer 1 API error: %s | session=%s", exc, session_id)
        raise
