"""Natural-language question templates for symptom tokens.

Maps canonical symptom token strings (as they appear in the 132-feature
training vocabulary) to patient-facing yes/no question text.  Used by the
dialogue service instead of raw ``"Do you have <token>?"`` strings.

All tokens are from the Kaggle "Disease Prediction Using Machine Learning"
dataset vocabulary.  A safe default fallback is provided for any token not
explicitly listed.
"""

from __future__ import annotations

QUESTION_TEMPLATES: dict[str, str] = {
    # ── Skin ──────────────────────────────────────────────────────────────
    "itching":                         "Are you experiencing itching on your skin?",
    "skin_rash":                       "Have you noticed any skin rash or unusual skin patches?",
    "nodal_skin_eruptions":            "Do you have nodular eruptions or bumps under the skin?",
    "skin_peeling":                    "Is your skin peeling or flaking?",
    "blister":                         "Do you have any blisters on your skin?",
    "silver_like_dusting":             "Is there a silver-like dusty coating on your skin?",
    "small_dents_in_nails":            "Do you have small dents or pits in your nails?",
    "inflammatory_nails":              "Are your nails inflamed, red, or swollen around the edges?",
    "pus_filled_pimples":              "Do you have pus-filled pimples or pustules?",
    "blackheads":                      "Do you have blackheads on your skin?",
    "scurring":                        "Do you notice scarring or scurring of the skin?",
    "red_sore_around_nose":            "Do you have red sores or crusting around your nose?",
    "yellow_crust_ooze":               "Is there a yellow crust or ooze on any skin areas?",
    "dischromic _patches":             "Do you have discoloured or patchy areas of skin?",
    # ── Eyes ──────────────────────────────────────────────────────────────
    "redness_of_eyes":                 "Are your eyes red or bloodshot?",
    "watering_from_eyes":              "Are your eyes watering excessively?",
    "blurred_and_distorted_vision":    "Is your vision blurry or distorted?",
    "visual_disturbances":             "Are you experiencing any visual disturbances or blind spots?",
    "yellowing_of_eyes":               "Has the white part of your eyes turned yellow?",
    # ── Head / Neurological ───────────────────────────────────────────────
    "headache":                        "Do you have a headache?",
    "dizziness":                       "Are you feeling dizzy or lightheaded?",
    "loss_of_balance":                 "Are you having trouble with your balance or coordination?",
    "stiff_neck":                      "Is your neck stiff or painful to move?",
    "neck_pain":                       "Do you have pain in your neck?",
    "spinning_movements":              "Do you experience a spinning sensation (vertigo)?",
    "unsteadiness":                    "Do you feel unsteady on your feet?",
    "weakness_of_one_body_side":       "Do you have weakness on one side of your body?",
    "slurred_speech":                  "Is your speech slurred or difficult to understand?",
    "altered_sensorium":               "Are you feeling confused or disoriented?",
    "lack_of_concentration":           "Are you having difficulty concentrating?",
    "loss_of_smell":                   "Have you lost your sense of smell?",
    "depression":                      "Are you feeling persistently depressed or hopeless?",
    "irritability":                    "Are you feeling unusually irritable or restless?",
    "restlessness":                    "Are you feeling restless or unable to relax?",
    "anxiety":                         "Are you experiencing anxiety or excessive worry?",
    "mood_swings":                     "Are you experiencing mood swings?",
    # ── Fever / Systemic ──────────────────────────────────────────────────
    "high_fever":                      "Are you experiencing a high fever?",
    "mild_fever":                      "Do you have a mild or low-grade fever?",
    "chills":                          "Are you having chills or feeling unusually cold?",
    "sweating":                        "Are you sweating excessively?",
    "shivering":                       "Are you shivering or trembling?",
    "malaise":                         "Are you feeling a general sense of unwellness or discomfort?",
    "fatigue":                         "Are you feeling unusually tired or weak?",
    "lethargy":                        "Do you feel lethargic or have very low energy?",
    # ── Respiratory ───────────────────────────────────────────────────────
    "cough":                           "Do you have a cough?",
    "breathlessness":                  "Are you having difficulty breathing or shortness of breath?",
    "chest_pain":                      "Do you feel pain or tightness in your chest?",
    "phlegm":                          "Are you coughing up phlegm or mucus?",
    "mucoid_sputum":                   "Are you producing mucoid (jelly-like) sputum?",
    "rusty_sputum":                    "Is your sputum rusty or blood-tinged?",
    "blood_in_sputum":                 "Is there blood in your sputum or phlegm?",
    "throat_irritation":               "Do you have a sore or irritated throat?",
    "sinus_pressure":                  "Do you feel pressure or pain in your sinuses?",
    "runny_nose":                      "Do you have a runny nose?",
    "congestion":                      "Are you feeling congested or stuffy?",
    "continuous_sneezing":             "Are you sneezing continuously?",
    "patches_in_throat":               "Do you have patches or white spots in your throat?",
    "fast_heart_rate":                 "Is your heart beating unusually fast?",
    "palpitations":                    "Are you experiencing palpitations or irregular heartbeat?",
    # ── Gastrointestinal ──────────────────────────────────────────────────
    "nausea":                          "Are you feeling nauseous?",
    "vomiting":                        "Have you vomited recently?",
    "abdominal_pain":                  "Do you have pain in your abdomen or stomach area?",
    "stomach_pain":                    "Do you have stomach pain or cramps?",
    "belly_pain":                      "Do you have pain in your belly area?",
    "diarrhoea":                       "Are you experiencing loose stools or diarrhoea?",
    "constipation":                    "Are you experiencing constipation or difficulty passing stool?",
    "indigestion":                     "Are you experiencing indigestion or heartburn?",
    "acidity":                         "Do you have acidity, acid reflux, or a burning sensation in your chest?",
    "loss_of_appetite":                "Have you lost your appetite or interest in eating?",
    "dehydration":                     "Are you feeling dehydrated or excessively thirsty?",
    "passage_of_gases":                "Are you passing an unusual amount of gas?",
    "internal_itching":                "Do you feel itching inside your body (e.g. in the gut)?",
    "ulcers_on_tongue":                "Do you have ulcers or sores on your tongue?",
    "pain_during_bowel_movements":     "Do you experience pain during bowel movements?",
    "pain_in_anal_region":             "Do you have pain in the anal region?",
    "bloody_stool":                    "Have you noticed blood in your stool?",
    "irritation_in_anus":              "Do you have irritation or itching in the anal area?",
    "stomach_bleeding":                "Have you had any bleeding from the stomach?",
    "distention_of_abdomen":           "Is your abdomen distended or unusually swollen?",
    "swelling_of_stomach":             "Is your stomach noticeably swollen?",
    # ── Liver / Jaundice ──────────────────────────────────────────────────
    "yellowish_skin":                  "Has your skin or the whites of your eyes turned yellow?",
    "dark_urine":                      "Is your urine darker than usual (tea-coloured)?",
    "yellow_urine":                    "Is your urine yellow or concentrated?",
    "acute_liver_failure":             "Have you been diagnosed with or suspected of acute liver failure?",
    "fluid_overload":                  "Are you retaining excess fluid (oedema / swelling)?",
    "fluid_overload.1":                "Do you have signs of significant fluid overload?",
    "history_of_alcohol_consumption":  "Do you have a history of heavy alcohol consumption?",
    # ── Musculoskeletal ───────────────────────────────────────────────────
    "joint_pain":                      "Are your joints painful or aching?",
    "swelling_joints":                 "Are your joints visibly swollen?",
    "movement_stiffness":              "Do you experience stiffness when moving your joints?",
    "muscle_pain":                     "Do you have muscle aches or body pain?",
    "muscle_weakness":                 "Do you feel weakness in your muscles?",
    "muscle_wasting":                  "Have you noticed any muscle wasting or loss of muscle mass?",
    "back_pain":                       "Do you have pain in your back?",
    "knee_pain":                       "Do you have pain in your knee(s)?",
    "hip_joint_pain":                  "Do you have pain in your hip joints?",
    "painful_walking":                 "Is walking painful for you?",
    "weakness_in_limbs":               "Do you feel weakness in your arms or legs?",
    "cramps":                          "Are you experiencing cramps?",
    # ── Urinary ───────────────────────────────────────────────────────────
    "burning_micturition":             "Do you feel a burning sensation when urinating?",
    "spotting_ urination":             "Are you experiencing spotting or irregular urination?",
    "spotting_urination":              "Are you experiencing spotting or irregular urination?",
    "continuous_feel_of_urine":        "Do you have a continuous urge to urinate?",
    "bladder_discomfort":              "Do you feel discomfort in your bladder?",
    "foul_smell_of urine":             "Does your urine have an unusually foul smell?",
    "foul_smell_of_urine":             "Does your urine have an unusually foul smell?",
    "polyuria":                        "Are you urinating much more frequently than normal?",
    # ── Weight / Metabolic ────────────────────────────────────────────────
    "weight_loss":                     "Have you lost weight recently without trying?",
    "weight_gain":                     "Have you gained weight recently without a clear reason?",
    "obesity":                         "Are you significantly overweight or obese?",
    "excessive_hunger":                "Are you feeling excessively hungry?",
    "increased_appetite":              "Has your appetite increased significantly?",
    "irregular_sugar_level":           "Do you have irregular blood sugar levels?",
    # ── Vascular / Circulatory ────────────────────────────────────────────
    "swollen_legs":                    "Are your legs swollen?",
    "swollen_blood_vessels":           "Do you have visibly swollen blood vessels?",
    "prominent_veins_on_calf":         "Do you notice prominent, bulging veins on your calf?",
    "bruising":                        "Are you bruising easily or noticing unexplained bruises?",
    "puffy_face_and_eyes":             "Is your face or the area around your eyes puffy or swollen?",
    "swollen_extremeties":             "Are your hands or feet swollen?",
    "swelled_lymph_nodes":             "Do you have swollen lymph nodes?",
    # ── Thyroid / Endocrine ───────────────────────────────────────────────
    "enlarged_thyroid":                "Do you have an enlarged thyroid (goitre) or neck swelling?",
    "brittle_nails":                   "Are your nails brittle or breaking easily?",
    "cold_hands_and_feets":            "Do you experience cold hands and feet?",
    "drying_and_tingling_lips":        "Are your lips dry and do you feel a tingling sensation?",
    "sunken_eyes":                     "Do your eyes look sunken?",
    # ── Infectious disease specific ───────────────────────────────────────
    "pain_behind_the_eyes":            "Do you have pain behind or around your eyes?",
    "toxic_look_(typhos)":             "Do you have a toxic or very ill appearance (typhos look)?",
    "red_spots_over_body":             "Do you have red spots distributed over your body?",
    "abnormal_menstruation":           "Are you experiencing abnormal menstruation?",
    # ── High-risk / exposure ──────────────────────────────────────────────
    "extra_marital_contacts":          "Have you had unprotected sexual contact outside a regular partnership?",
    "receiving_blood_transfusion":     "Have you recently received a blood transfusion?",
    "receiving_unsterile_injections":  "Have you received any injections with unsterile equipment?",
    "family_history":                  "Is there a family history of this type of condition?",
    # ── Severe / critical ─────────────────────────────────────────────────
    "coma":                            "Have you experienced or been close to losing consciousness?",
    "loss_of_appetite":                "Have you lost your appetite or interest in eating?",
}


def get_question(symptom: str) -> str:
    """Return a natural-language yes/no question for the given symptom token.

    First looks up the exact token in ``QUESTION_TEMPLATES``.  Falls back to
    a generic template that humanises the underscore-separated token.

    Args:
        symptom: Canonical symptom token (e.g. ``"high_fever"``).

    Returns:
        A patient-facing question string ending with ``"?"`` ready to display
        in the chat UI.  The response is always expected to be yes or no.
    """
    if symptom in QUESTION_TEMPLATES:
        return QUESTION_TEMPLATES[symptom]
    # Generic fallback: replace underscores and title-case.
    readable = symptom.replace("_", " ")
    return f"Are you experiencing {readable}? (yes / no)"
