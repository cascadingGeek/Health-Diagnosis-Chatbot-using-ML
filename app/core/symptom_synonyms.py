"""Synonym map for colloquial symptom descriptions.

Maps common patient-facing phrases to canonical symptom tokens from the
132-feature training vocabulary.  Used as a pre-processing step before
fuzzy matching so that everyday language lands on the right token even
when rapidfuzz alone would score poorly.

Layer 1 (Claude) is the primary synonym resolver for free text.  This
module acts as a deterministic fallback for the dialogue service and any
other code that needs fast, offline synonym resolution.
"""

from __future__ import annotations

# Keys: lowercase normalised colloquial phrases.
# Values: canonical symptom tokens from the training vocabulary.
SYNONYM_MAP: dict[str, str] = {
    # General / fatigue
    "tired": "fatigue",
    "exhausted": "fatigue",
    "no energy": "fatigue",
    "sluggish": "lethargy",
    "weak": "fatigue",
    "weakness": "fatigue",
    "feeling weak": "fatigue",
    "run down": "fatigue",

    # Fever / temperature
    "temperature": "high_fever",
    "hot": "high_fever",
    "burning up": "high_fever",
    "low grade fever": "mild_fever",
    "mild temperature": "mild_fever",
    "shaking": "shivering",
    "trembling": "shivering",
    "cold sweat": "sweating",
    "night sweats": "sweating",

    # Pain
    "tummy ache": "stomach_pain",
    "belly ache": "stomach_pain",
    "stomach ache": "stomach_pain",
    "gut pain": "abdominal_pain",
    "stomach cramps": "stomach_pain",
    "period pain": "abdominal_pain",
    "lower back pain": "back_pain",
    "sore back": "back_pain",
    "sore throat": "throat_irritation",
    "throat pain": "throat_irritation",
    "chest tightness": "chest_pain",
    "heart pain": "chest_pain",
    "leg pain": "muscle_pain",
    "arm pain": "muscle_pain",
    "body ache": "muscle_pain",
    "body pain": "muscle_pain",
    "sore muscles": "muscle_pain",
    "leg cramps": "cramps",
    "muscle cramps": "cramps",
    "anal pain": "pain_in_anal_region",
    "rectal pain": "pain_in_anal_region",
    "butt pain": "pain_in_anal_region",
    "groin pain": "hip_joint_pain",
    "knee ache": "knee_pain",
    "eye pain": "pain_behind_the_eyes",
    "headache behind eyes": "pain_behind_the_eyes",

    # Gastrointestinal
    "throwing up": "vomiting",
    "puking": "vomiting",
    "sick to my stomach": "nausea",
    "feel like vomiting": "nausea",
    "queasy": "nausea",
    "loose stool": "diarrhoea",
    "loose stools": "diarrhoea",
    "runny stool": "diarrhoea",
    "watery stool": "diarrhoea",
    "bloody poo": "bloody_stool",
    "blood in stool": "bloody_stool",
    "blood in poo": "bloody_stool",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "bloating": "distention_of_abdomen",
    "bloated": "distention_of_abdomen",
    "not hungry": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "no thirst": "dehydration",
    "very thirsty": "dehydration",
    "gassy": "passage_of_gases",
    "flatulence": "passage_of_gases",
    "burping": "indigestion",
    "belching": "indigestion",
    "constipated": "constipation",
    "cant poop": "constipation",

    # Respiratory
    "short of breath": "breathlessness",
    "out of breath": "breathlessness",
    "cant breathe": "breathlessness",
    "difficulty breathing": "breathlessness",
    "dry cough": "cough",
    "wet cough": "phlegm",
    "coughing blood": "blood_in_sputum",
    "blood when coughing": "blood_in_sputum",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "runny nose": "runny_nose",
    "sneezing": "continuous_sneezing",
    "sinus": "sinus_pressure",

    # Skin
    "rash": "skin_rash",
    "itchy skin": "itching",
    "skin itch": "itching",
    "hives": "skin_rash",
    "yellow skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes",
    "peeling skin": "skin_peeling",
    "pimples": "pus_filled_pimples",
    "spots": "red_spots_over_body",

    # Eyes / vision
    "red eyes": "redness_of_eyes",
    "pink eye": "redness_of_eyes",
    "blurry vision": "blurred_and_distorted_vision",
    "blurred vision": "blurred_and_distorted_vision",
    "watery eyes": "watering_from_eyes",
    "teary eyes": "watering_from_eyes",
    "sunken eyes": "sunken_eyes",
    "puffy eyes": "puffy_face_and_eyes",

    # Neurological / mental
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "spinning": "spinning_movements",
    "vertigo": "spinning_movements",
    "brain fog": "lack_of_concentration",
    "cant concentrate": "lack_of_concentration",
    "forgetful": "lack_of_concentration",
    "confused": "altered_sensorium",
    "disoriented": "altered_sensorium",
    "anxious": "anxiety",
    "worried": "anxiety",
    "sad": "depression",
    "depressed": "depression",
    "moody": "mood_swings",
    "irritable": "irritability",
    "restless": "restlessness",
    "cant sleep": "restlessness",
    "speech problems": "slurred_speech",
    "cant talk properly": "slurred_speech",
    "loss of balance": "loss_of_balance",
    "unstable walking": "unsteadiness",
    "one side weak": "weakness_of_one_body_side",

    # Urinary
    "burning pee": "burning_micturition",
    "painful urination": "burning_micturition",
    "frequent urination": "polyuria",
    "peeing a lot": "polyuria",
    "dark pee": "dark_urine",
    "dark urine": "dark_urine",
    "smelly urine": "foul_smell_of_urine",
    "cloudy urine": "foul_smell_of_urine",
    "blood in urine": "spotting_urination",
    "urge to pee": "continuous_feel_of_urine",
    "bladder pain": "bladder_discomfort",

    # Weight / metabolic
    "losing weight": "weight_loss",
    "gained weight": "weight_gain",
    "always hungry": "excessive_hunger",
    "thirsty all the time": "excessive_hunger",
    "overweight": "obesity",
    "obese": "obesity",

    # Swelling
    "swollen legs": "swollen_legs",
    "swollen feet": "swollen_extremeties",
    "swollen hands": "swollen_extremeties",
    "swollen face": "puffy_face_and_eyes",
    "swollen neck": "enlarged_thyroid",
    "swollen joints": "swelling_joints",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "lumps in neck": "swelled_lymph_nodes",
}


def resolve_synonym(phrase: str) -> str | None:
    """Look up a colloquial phrase in the synonym map.

    Args:
        phrase: User-supplied symptom description (case-insensitive).

    Returns:
        Canonical symptom token if found, otherwise ``None``.
    """
    return SYNONYM_MAP.get(phrase.strip().lower())
