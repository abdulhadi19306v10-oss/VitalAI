"""Canonical medical synonym, US/UK spelling, and Kaggle typo alias mappings."""

from __future__ import annotations

import re

# Single source of truth: canonical Kaggle symptom key -> clinical & lay aliases.
CANONICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "itching": ("pruritus", "itchy_skin", "itch"),
    "skin_rash": ("rash", "exanthem", "eruption"),
    "nodal_skin_eruptions": ("skin_nodules", "bumps_on_skin"),
    "continuous_sneezing": ("sneezing", "rhinitis"),
    "shivering": ("rigors", "trembling"),
    "chills": ("feeling_cold", "chilly"),
    "joint_pain": ("arthralgia", "aching_joints"),
    "stomach_pain": ("gastric_pain", "epigastric_pain"),
    "acidity": ("heartburn", "acid_reflux", "pyrosis"),
    "ulcers_on_tongue": ("tongue_ulcer", "mouth_ulcer", "canker_sore"),
    "muscle_wasting": ("muscle_atrophy", "sarcopenia"),
    "vomiting": ("emesis", "throwing_up"),
    "burning_micturition": (
        "dysuria",
        "painful_urination",
        "burning_urine",
        "pain_while_urinating",
    ),
    "spotting_urination": ("urinary_spotting", "urethral_spotting"),
    "fatigue": ("tiredness", "exhaustion", "asthenia"),
    "weight_gain": ("gaining_weight",),
    "anxiety": ("nervousness", "anxious"),
    "cold_hands_and_feets": ("cold_hands_and_feet", "cold_extremities"),
    "mood_swings": ("labile_mood", "emotional_lability"),
    "weight_loss": ("unintended_weight_loss", "losing_weight"),
    "restlessness": ("agitation", "akathisia"),
    "lethargy": ("sluggishness", "drowsiness"),
    "patches_in_throat": ("throat_patches", "exudative_pharyngitis", "white_patches_throat"),
    "irregular_sugar_level": ("blood_sugar_fluctuation", "hyperglycemia", "dysglycemia"),
    "cough": ("coughing", "tussis"),
    "high_fever": ("pyrexia", "fever", "high_temperature", "hyperthermia"),
    "sunken_eyes": ("hollow_eyes", "enophthalmos"),
    "breathlessness": (
        "dyspnea",
        "dyspnoea",
        "shortness_of_breath",
        "sob",
        "difficulty_breathing",
        "air_hunger",
    ),
    "sweating": ("diaphoresis", "perspiration", "sweats", "hyperhidrosis"),
    "dehydration": ("fluid_loss", "volume_depletion"),
    "indigestion": ("dyspepsia", "upset_stomach"),
    "headache": ("cephalalgia", "cephalgia", "head_pain"),
    "yellowish_skin": ("jaundice_skin", "icterus", "yellow_skin"),
    "dark_urine": ("tea_colored_urine", "bilirubinuria"),
    "nausea": ("queasiness", "feeling_sick"),
    "loss_of_appetite": ("anorexia", "poor_appetite"),
    "pain_behind_the_eyes": ("retro_orbital_pain", "eye_socket_pain"),
    "back_pain": ("dorsalgia", "lumbar_pain", "lumbago"),
    "constipation": ("hard_stools", "infrequent_bowels"),
    "abdominal_pain": ("belly_ache", "tummy_pain", "abdominal_cramp"),
    "diarrhoea": ("diarrhea", "loose_stools", "watery_stools"),
    "mild_fever": ("low_grade_fever", "subfebrile"),
    "yellow_urine": ("amber_urine",),
    "yellowing_of_eyes": ("scleral_icterus", "yellow_eyes", "icteric_sclera"),
    "acute_liver_failure": ("hepatic_failure", "fulminant_hepatitis"),
    "fluid_overload": ("hypervolemia", "fluid_retention"),
    "swelling_of_stomach": ("ascites", "abdominal_swelling"),
    "swelled_lymph_nodes": ("lymphadenopathy", "swollen_glands", "enlarged_lymph_nodes"),
    "malaise": ("general_discomfort", "feeling_unwell"),
    "blurred_and_distorted_vision": ("blurry_vision", "blurred_vision", "visual_blurring"),
    "phlegm": ("sputum", "mucus"),
    "throat_irritation": ("sore_throat", "pharyngitis", "scratchy_throat"),
    "redness_of_eyes": ("conjunctivitis", "bloodshot_eyes", "red_eyes"),
    "sinus_pressure": ("sinus_pain", "facial_pressure"),
    "runny_nose": ("rhinorrhea", "nasal_discharge"),
    "congestion": ("nasal_congestion", "stuffy_nose"),
    "chest_pain": ("angina", "thoracic_pain", "chest_tightness"),
    "weakness_in_limbs": ("limb_weakness", "extremity_weakness", "paresis"),
    "fast_heart_rate": ("tachycardia", "rapid_pulse", "racing_heart"),
    "pain_during_bowel_movements": ("dyschezia", "painful_defecation"),
    "pain_in_anal_region": ("proctalgia", "anal_pain"),
    "bloody_stool": ("hematochezia", "rectal_bleeding", "blood_in_stool"),
    "irritation_in_anus": ("pruritus_ani", "anal_itching"),
    "neck_pain": ("cervicalgia", "cervical_pain"),
    "dizziness": ("lightheadedness", "presyncope", "giddiness"),
    "cramps": ("muscle_cramps", "spasms"),
    "bruising": ("ecchymosis", "contusion"),
    "obesity": ("high_bmi", "overweight"),
    "swollen_legs": ("leg_edema", "pedal_edema", "peripheral_edema"),
    "swollen_blood_vessels": ("varicosities", "engorged_veins"),
    "puffy_face_and_eyes": ("periorbital_edema", "facial_edema", "puffy_eyes"),
    "enlarged_thyroid": ("goiter", "goitre", "thyromegaly"),
    "brittle_nails": ("onychoschizia", "fragile_nails"),
    "swollen_extremeties": ("swollen_extremities", "hand_and_foot_edema", "acral_edema"),
    "excessive_hunger": ("polyphagia", "hyperphagia"),
    "extra_marital_contacts": ("unprotected_sexual_exposure", "high_risk_sexual_contact"),
    "drying_and_tingling_lips": ("paresthesia_lips", "dry_tingling_lips"),
    "slurred_speech": ("dysarthria", "speech_slurring"),
    "knee_pain": ("gonalgia", "knee_arthralgia"),
    "hip_joint_pain": ("coxalgia", "hip_pain"),
    "muscle_weakness": ("myasthenia", "proximal_weakness"),
    "stiff_neck": ("nuchal_rigidity", "neck_stiffness"),
    "swelling_joints": ("joint_swelling", "joint_effusion"),
    "movement_stiffness": ("joint_stiffness", "morning_stiffness"),
    "spinning_movements": ("vertigo", "rotational_dizziness"),
    "loss_of_balance": ("ataxia", "imbalance", "disequilibrium"),
    "unsteadiness": ("gait_instability", "postural_instability"),
    "weakness_of_one_body_side": (
        "hemiparesis",
        "hemiplegia",
        "unilateral_weakness",
        "one_sided_weakness",
    ),
    "loss_of_smell": ("anosmia", "hyposmia"),
    "bladder_discomfort": ("suprapubic_discomfort", "pelvic_bladder_pain"),
    "foul_smell_of_urine": ("malodorous_urine", "foul_urine"),
    "continuous_feel_of_urine": ("urinary_urgency", "frequent_urge_to_urinate"),
    "passage_of_gases": ("flatulence", "flatus"),
    "internal_itching": ("visceral_pruritus",),
    "toxic_look_(typhos)": ("toxic_appearance", "toxic_look_typhus"),
    "depression": ("depressed_mood", "low_mood"),
    "irritability": ("agitation_mood",),
    "muscle_pain": ("myalgia", "body_ache", "body_aches", "sore_muscles"),
    "altered_sensorium": (
        "confusion",
        "delirium",
        "altered_mental_status",
        "encephalopathy",
    ),
    "red_spots_over_body": ("petechiae", "maculopapular_rash", "erythematous_spots"),
    "belly_pain": ("periumbilical_pain", "stomach_cramps"),
    "abnormal_menstruation": ("menstrual_irregularity", "amenorrhea", "menorrhagia"),
    "dischromic_patches": ("hypopigmented_patches", "skin_discoloration"),
    "watering_from_eyes": ("epiphora", "lacrimation", "tearing_eyes", "watery_eyes"),
    "increased_appetite": ("hyperorexia", "increased_hunger"),
    "polyuria": ("frequent_urination", "excessive_urination"),
    "family_history": ("hereditary_history", "genetic_predisposition"),
    "mucoid_sputum": ("mucus_sputum", "clear_sputum"),
    "rusty_sputum": ("rust_colored_sputum", "brownish_sputum"),
    "lack_of_concentration": ("poor_concentration", "brain_fog"),
    "visual_disturbances": ("scotoma", "diplopia", "photopsia"),
    "receiving_blood_transfusion": ("blood_transfusion_history",),
    "receiving_unsterile_injections": ("unsterile_needle_exposure", "iv_needle_exposure"),
    "coma": ("unconsciousness", "unresponsive", "loss_of_consciousness"),
    "stomach_bleeding": ("gi_bleeding", "hematemesis", "upper_gi_bleed"),
    "distention_of_abdomen": ("abdominal_distention", "abdominal_bloating"),
    "history_of_alcohol_consumption": ("alcohol_use", "alcoholism", "heavy_drinking"),
    "blood_in_sputum": (
        "hemoptysis",
        "haemoptysis",
        "coughing_blood",
        "coughing_up_blood",
    ),
    "prominent_veins_on_calf": ("calf_varicosities", "dilated_calf_veins"),
    "palpitations": ("heart_palpitations", "fluttering_heart"),
    "painful_walking": ("claudication", "antalgic_gait"),
    "pus_filled_pimples": ("pustules", "purulent_acne"),
    "blackheads": ("comedones", "open_comedones"),
    "scurring": ("acne_scarring", "skin_scarring", "scarring"),
    "skin_peeling": ("desquamation", "flaking_skin"),
    "silver_like_dusting": ("silvery_scales", "psoriatic_scales"),
    "small_dents_in_nails": ("nail_pitting",),
    "inflammatory_nails": ("paronychia", "nail_inflammation", "psoriatic_nails"),
    "blister": ("vesicles", "bullae", "skin_blisters"),
    "red_sore_around_nose": ("perinasal_sores", "nasal_impetigo"),
    "yellow_crust_ooze": ("honey_colored_crust", "crusting_exudate"),
}

# Clean up Kaggle typos for clinician-facing display while preserving canonical ML keys.
DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "cold_hands_and_feets": "Cold Hands And Feet",
    "diarrhoea": "Diarrhoea (Diarrhea)",
    "scurring": "Scarring (Scurring)",
    "swollen_extremeties": "Swollen Extremities",
    "toxic_look_(typhos)": "Toxic Look (Typhus)",
}


def _build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in CANONICAL_SYNONYMS.items():
        for alias in aliases:
            if alias in lookup and lookup[alias] != canonical:
                raise ValueError(
                    f"Duplicate alias '{alias}' mapped to both "
                    f"'{lookup[alias]}' and '{canonical}'."
                )
            lookup[alias] = canonical
    return lookup


SYMPTOM_ALIASES: dict[str, str] = _build_alias_lookup()


def normalize_token(raw: str) -> str:
    """Convert raw text or symptom key into normalized snake_case."""
    cleaned = re.sub(r"[\s\-]+", "_", str(raw).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def resolve_symptom_key(raw: str) -> str:
    """Normalize a symptom name or clinical synonym to its canonical Kaggle feature key."""
    norm = normalize_token(raw)
    return SYMPTOM_ALIASES.get(norm, norm)


def get_aliases_for_symptom(canonical_key: str) -> list[str]:
    """Return human-readable alias labels for UI search matching."""
    aliases = CANONICAL_SYNONYMS.get(canonical_key, ())
    return [a.replace("_", " ") for a in aliases]
