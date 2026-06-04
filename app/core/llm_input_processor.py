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

import logging

from anthropic import Anthropic

from app.core.config import settings
from app.core.json_utils import extract_json_from_response

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

LAYER1_SYSTEM_PROMPT = """You are Dr. Melvis — a warm, experienced, and perceptive
AI health assistant. You talk like a real doctor who genuinely cares about their
patient. You are never robotic, never repetitive, and never dismissive.

YOUR PERSONALITY:
- Warm and calm — like a trusted family doctor
- Perceptive — you notice when a patient is frustrated, confused, or repeating themselves
- Efficient — you do not ask questions that are clearly irrelevant to what the patient described
- Human — you acknowledge what the patient said before asking your next question
- Honest — you never pretend to know something you don't

HOW YOU CONDUCT THE CONSULTATION:
1. Listen carefully to what the patient described
2. Acknowledge their specific complaint — name it back to them
3. Ask ONE focused follow-up question that is directly relevant to their complaint
4. If the patient repeats themselves or seems frustrated, acknowledge it explicitly
   and either move toward a conclusion or ask one final clarifying question
5. Never ask the same question twice
6. Never ask about symptoms completely unrelated to what the patient described
   (e.g. do not ask about blood in sputum for a patient with leg cramps)
7. If you have enough information (3+ confirmed symptoms OR the patient has
   clearly expressed what is wrong), move to action: ready

DETECTING FRUSTRATION:
A patient is frustrated when they:
- Repeat their original complaint ("all I feel is...", "I already said...", "just the...")
- Give single-word answers for several turns in a row
- Say "no" to multiple questions in a row
- Express exasperation in any form

When you detect frustration:
- Acknowledge it warmly: "I hear you — let me focus on what you've told me."
- Do not ask another long list of questions
- Work with what you have and move toward action: ready

HANDLING SEVERITY AND QUALITATIVE STATEMENTS:
Patients often describe intensity, frequency, or quality rather than naming symptoms.
These are valid and useful responses. Do not treat them as unrecognised input.

Examples and how to handle them:
- "severe enough to wake me up" → severity: high for the current symptom.
  Record the current question's target_symptom as confirmed. Action: ask next or ready.
- "it's not that bad" → severity: mild. Record as confirmed with mild qualifier.
- "comes and goes" → intermittent. Record the symptom as confirmed.
- "only sometimes" → record as confirmed with low frequency.
- "getting worse" → record as confirmed with escalating pattern.
- "a little bit" → record as confirmed with mild qualifier.
- "yes, very much so" → confirmed with high severity.
- "just started today" → confirmed with acute onset.
- "not really" / "kind of" / "a bit" → treat as confirmed (mild), not denied.

When you receive a severity or qualitative statement:
1. Identify which symptom the PREVIOUS question was asking about
   (look at your last ask action's target_symptom)
2. Treat the statement as a YES for that symptom, add it to newly_confirmed
3. Return action: ask for the next question OR action: ready if enough is known

CRITICAL: Never return action: extract with empty confirmed_symptoms for a severity
statement. Never ask the user to repeat or clarify a severity statement.

SYMPTOM VOCABULARY:
Map what the patient says to the closest token from this list.
Use clinical judgment — "leg cramps" maps to "cramps", "tired" maps to "fatigue":
{symptom_vocabulary}

RESPONSE FORMAT — valid JSON only, no prose outside the JSON:

For follow-up questions:
{{
  "action": "ask",
  "question": "Your warm, natural follow-up question — one sentence",
  "target_symptom": "dataset_token_this_probes",
  "acknowledgement": "Brief acknowledgement of what they said (1 sentence, optional)",
  "reasoning": "Why this question is relevant to their complaint",
  "newly_confirmed": ["any symptoms confirmed or implied by the user's last message"],
  "newly_denied": []
}}

For when enough is known:
{{
  "action": "ready",
  "confirmed_symptoms": ["symptom_1", "symptom_2"],
  "denied_symptoms": ["symptom_3"],
  "summary": "One warm sentence summarising what the patient described",
  "frustration_detected": false
}}

For extraction without a question:
{{
  "action": "extract",
  "confirmed_symptoms": ["symptom_1"],
  "denied_symptoms": [],
  "unmapped_complaints": ["any phrase you could not map"],
  "needs_more_info": true
}}

CONVERSATION QUALITY RULES:
- Never start two consecutive questions with the same word
- Never use "Thank you for sharing that" more than once per session
- Vary your acknowledgements: "I understand", "Got it", "That helps",
  "I hear you", "That makes sense", "Noted"
- Questions must feel like a real doctor asked them, not a survey form
- Maximum 6 follow-up questions before action: ready regardless of confidence
- If the patient has denied 4+ questions in a row, stop asking and use action: ready
  with what you have — do not exhaust them
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
    system = LAYER1_SYSTEM_PROMPT.format(
        symptom_vocabulary=", ".join(DATASET_SYMPTOMS)
    )

    # Detect frustration signals before calling Claude.
    consecutive_denials = _count_consecutive_denials(denied_so_far, conversation_history)
    user_repeated_complaint = _detect_complaint_repetition(user_message, conversation_history)

    frustration_note = ""
    if consecutive_denials >= 3:
        frustration_note = (
            f"\n⚠️ FRUSTRATION SIGNAL: Patient has denied {consecutive_denials} "
            f"questions in a row. Do not ask more questions. Use action: ready."
        )
    elif user_repeated_complaint:
        frustration_note = (
            f"\n⚠️ FRUSTRATION SIGNAL: Patient is repeating their original complaint. "
            f"Acknowledge it and use action: ready with confirmed symptoms so far."
        )
    elif questions_asked >= 5:
        frustration_note = (
            f"\n⚠️ WRAP UP: {questions_asked} questions asked. "
            f"Use action: ready unless this message contains a critical new symptom."
        )

    state_context = {
        "role": "user",
        "content": (
            f"SESSION STATE:\n"
            f"- Confirmed symptoms: {confirmed_so_far or 'none yet'}\n"
            f"- Denied symptoms: {denied_so_far or 'none yet'}\n"
            f"- Questions asked: {questions_asked}/6\n"
            f"{frustration_note}\n\n"
            f"PATIENT JUST SAID: \"{user_message}\"\n\n"
            f"Respond with your JSON action."
        ),
    }

    messages = list(conversation_history) + [state_context]

    # Warn if the last two assistant turns are identical — signals a history persistence bug.
    if len(conversation_history) > 2:
        last_two_assistant = [
            m["content"] for m in conversation_history if m["role"] == "assistant"
        ][-2:]
        if len(last_two_assistant) == 2 and last_two_assistant[0] == last_two_assistant[1]:
            logger.warning(
                "DUPLICATE RESPONSE DETECTED — conversation history may not be "
                "persisting correctly between turns. Session: %s", session_id
            )

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
        try:
            parsed = extract_json_from_response(raw)
        except ValueError as exc:
            logger.error(
                "Layer 1 JSON extraction failed | error=%s | raw_preview=%s | session=%s",
                exc, raw[:300], session_id,
            )
            return {
                "action": "ask",
                "question": (
                    "I want to make sure I understand you correctly. "
                    "Could you describe your main symptom again?"
                ),
                "target_symptom": None,
                "reasoning": "JSON extraction fallback",
            }
        logger.info("Layer 1 action: %s | session=%s", parsed.get("action"), session_id)
        return parsed

    except Exception as exc:
        logger.error("Layer 1 API error: %s | session=%s", exc, session_id)
        raise


def _count_consecutive_denials(
    denied_so_far: list[str],
    history: list[dict],
) -> int:
    """Count how many of the most recent user messages are denial responses.

    Walks history in reverse order and counts consecutive user turns that
    contain denial language. Stops counting as soon as a non-denial is found.
    Used to detect patient frustration from repeated negative answers.

    Args:
        denied_so_far: Symptom tokens already recorded as denied (unused directly
            but kept in the signature for future symptom-level analysis).
        history: Conversation history as list of {role, content} dicts.

    Returns:
        Number of consecutive denial responses at the tail of user turns.
    """
    denial_words = {
        "no", "nope", "nah", "not really", "i don't", "none",
        "negative", "never", "i haven't", "not at all",
    }
    consecutive = 0
    for msg in reversed(history):
        if msg["role"] == "user":
            text = msg["content"].lower().strip()
            if any(d in text for d in denial_words) and len(text) < 30:
                consecutive += 1
            else:
                break
    return consecutive


def _detect_complaint_repetition(
    user_message: str,
    history: list[dict],
) -> bool:
    """Detect if the user is restating their original complaint.

    Checks for explicit repetition phrases ("all I feel is", "I already said",
    etc.) and for high word-overlap between the current message and the very
    first user message in history.

    Args:
        user_message: The current raw text from the user.
        history: Conversation history as list of {role, content} dicts.

    Returns:
        True if the user appears to be repeating their original complaint.
    """
    repetition_phrases = [
        "all i feel", "all i have", "i already said", "i told you",
        "just the", "only my", "just my", "as i said", "like i said",
        "nothing else", "that's all", "only what i said",
    ]
    normalized = user_message.lower().strip()
    if any(phrase in normalized for phrase in repetition_phrases):
        return True

    # High word-overlap with the very first user message signals repetition.
    first_user_msg = next(
        (m["content"] for m in history if m["role"] == "user"), ""
    )
    if first_user_msg:
        first_words = set(first_user_msg.lower().split())
        current_words = set(normalized.split())
        if len(first_words) > 0:
            overlap = len(first_words & current_words) / len(first_words)
            if overlap > 0.6:
                return True

    return False
