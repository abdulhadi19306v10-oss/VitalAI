"""Clinical knowledge base covering all 41 VitalAI diseases and red-flag rules."""

from __future__ import annotations

from dataclasses import dataclass

URGENCY_RANK: dict[str, int] = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4,
}


@dataclass(frozen=True)
class DiseaseProfile:
    """Domain ontology record for a classified disease."""

    canonical_name: str
    display_name: str
    category: str
    base_urgency: str
    specialist: str
    contagious: bool
    actions: tuple[str, ...]


DISEASE_KB: dict[str, DiseaseProfile] = {
    "(vertigo) Paroymsal Positional Vertigo": DiseaseProfile(
        canonical_name="(vertigo) Paroymsal Positional Vertigo",
        display_name="Benign Paroxysmal Positional Vertigo (BPPV)",
        category="neurological_vestibular",
        base_urgency="Moderate",
        specialist="ENT Specialist / Neurologist",
        contagious=False,
        actions=(
            "Sit or lie down immediately during spinning episodes to prevent falls.",
            "Avoid sudden head movements and driving until evaluated.",
        ),
    ),
    "AIDS": DiseaseProfile(
        canonical_name="AIDS",
        display_name="HIV / AIDS",
        category="immunological_infectious",
        base_urgency="High",
        specialist="Infectious Disease Specialist",
        contagious=True,
        actions=(
            "Seek prompt confirmatory laboratory testing (ELISA / PCR) and medical care.",
            "Avoid unprotected exposure and seek specialist antiretroviral management.",
        ),
    ),
    "Acne": DiseaseProfile(
        canonical_name="Acne",
        display_name="Acne Vulgaris",
        category="dermatological",
        base_urgency="Low",
        specialist="Dermatologist",
        contagious=False,
        actions=(
            "Cleanse affected skin gently with mild non-comedogenic cleanser.",
            "Avoid squeezing lesions to prevent scarring or secondary bacterial infection.",
        ),
    ),
    "Alcoholic hepatitis": DiseaseProfile(
        canonical_name="Alcoholic hepatitis",
        display_name="Alcoholic Hepatitis",
        category="hepatic",
        base_urgency="High",
        specialist="Gastroenterologist / Hepatologist",
        contagious=False,
        actions=(
            "Stop all alcohol consumption immediately.",
            "Seek urgent liver function testing and physician assessment.",
        ),
    ),
    "Allergy": DiseaseProfile(
        canonical_name="Allergy",
        display_name="Allergic Rhinitis / Allergy",
        category="immunological",
        base_urgency="Low",
        specialist="Allergist / General Physician",
        contagious=False,
        actions=(
            "Avoid suspected environmental or food triggers.",
            "Monitor for any facial swelling or breathing difficulty requiring urgent care.",
        ),
    ),
    "Arthritis": DiseaseProfile(
        canonical_name="Arthritis",
        display_name="Arthritis",
        category="musculoskeletal",
        base_urgency="Moderate",
        specialist="Rheumatologist",
        contagious=False,
        actions=(
            "Rest inflamed joints and avoid high-impact strain.",
            "Schedule rheumatological evaluation for targeted anti-inflammatory therapy.",
        ),
    ),
    "Bronchial Asthma": DiseaseProfile(
        canonical_name="Bronchial Asthma",
        display_name="Bronchial Asthma",
        category="respiratory",
        base_urgency="High",
        specialist="Pulmonologist",
        contagious=False,
        actions=(
            "Sit upright, keep calm, and use a prescribed rescue inhaler if available.",
            "Seek emergency care if speech becomes broken or lips turn bluish.",
        ),
    ),
    "Cervical spondylosis": DiseaseProfile(
        canonical_name="Cervical spondylosis",
        display_name="Cervical Spondylosis",
        category="musculoskeletal_neurological",
        base_urgency="Moderate",
        specialist="Orthopedist / Neurologist",
        contagious=False,
        actions=(
            "Maintain neutral neck posture and avoid heavy lifting.",
            "Seek evaluation promptly if limb numbness or progressive weakness occurs.",
        ),
    ),
    "Chicken pox": DiseaseProfile(
        canonical_name="Chicken pox",
        display_name="Chickenpox (Varicella)",
        category="infectious_viral",
        base_urgency="Moderate",
        specialist="General Physician",
        contagious=True,
        actions=(
            "Isolate from pregnant individuals, newborns, and immunocompromised people.",
            "Maintain hydration and avoid scratching vesicles to prevent infection.",
        ),
    ),
    "Chronic cholestasis": DiseaseProfile(
        canonical_name="Chronic cholestasis",
        display_name="Chronic Cholestasis",
        category="hepatic",
        base_urgency="High",
        specialist="Gastroenterologist / Hepatologist",
        contagious=False,
        actions=(
            "Schedule urgent hepatobiliary labs and abdominal imaging.",
            "Avoid over-the-counter hepatotoxic medications until evaluated.",
        ),
    ),
    "Common Cold": DiseaseProfile(
        canonical_name="Common Cold",
        display_name="Common Cold (Viral URI)",
        category="respiratory_viral",
        base_urgency="Low",
        specialist="General Physician",
        contagious=True,
        actions=(
            "Rest, stay well hydrated, and use warm saline rinses for nasal relief.",
            "Seek care if fever persists beyond 3 days or breathing worsens.",
        ),
    ),
    "Dengue": DiseaseProfile(
        canonical_name="Dengue",
        display_name="Dengue Fever",
        category="infectious_vector",
        base_urgency="High",
        specialist="Infectious Disease Specialist / Internal Medicine",
        contagious=False,
        actions=(
            "Obtain urgent CBC/platelet count and dengue antigen testing.",
            "Maintain oral hydration and avoid aspirin/NSAIDs due to bleeding risk.",
        ),
    ),
    "Diabetes": DiseaseProfile(
        canonical_name="Diabetes",
        display_name="Diabetes Mellitus",
        category="endocrine",
        base_urgency="Moderate",
        specialist="Endocrinologist",
        contagious=False,
        actions=(
            "Check fasting and post-meal blood glucose / HbA1c promptly.",
            "Maintain hydration and seek urgent care if persistent vomiting or confusion develops.",
        ),
    ),
    "Dimorphic hemmorhoids(piles)": DiseaseProfile(
        canonical_name="Dimorphic hemmorhoids(piles)",
        display_name="Hemorrhoids (Piles)",
        category="gastrointestinal",
        base_urgency="Low",
        specialist="General Surgeon / Gastroenterologist",
        contagious=False,
        actions=(
            "Increase dietary fiber, fluid intake, and warm sitz baths.",
            "Seek prompt evaluation if rectal bleeding is heavy or causes dizziness.",
        ),
    ),
    "Drug Reaction": DiseaseProfile(
        canonical_name="Drug Reaction",
        display_name="Adverse Drug Reaction",
        category="pharmacological_immunological",
        base_urgency="High",
        specialist="General Physician / Allergist",
        contagious=False,
        actions=(
            "Contact the prescribing clinician immediately before taking the next dose.",
            "Go to emergency care if rash spreads rapidly, blisters, or affects breathing.",
        ),
    ),
    "Fungal infection": DiseaseProfile(
        canonical_name="Fungal infection",
        display_name="Cutaneous Fungal Infection",
        category="dermatological",
        base_urgency="Low",
        specialist="Dermatologist",
        contagious=True,
        actions=(
            "Keep affected skin clean and dry; avoid sharing towels or clothing.",
            "Consult a doctor or pharmacist for targeted topical antifungal treatment.",
        ),
    ),
    "GERD": DiseaseProfile(
        canonical_name="GERD",
        display_name="Gastroesophageal Reflux Disease (GERD)",
        category="gastrointestinal",
        base_urgency="Low",
        specialist="Gastroenterologist",
        contagious=False,
        actions=(
            "Avoid lying flat within 2-3 hours after meals and limit trigger foods.",
            "Seek urgent evaluation if chest pain radiates or occurs with exertion.",
        ),
    ),
    "Gastroenteritis": DiseaseProfile(
        canonical_name="Gastroenteritis",
        display_name="Acute Gastroenteritis",
        category="gastrointestinal_infectious",
        base_urgency="Moderate",
        specialist="General Physician / Gastroenterologist",
        contagious=True,
        actions=(
            "Take small, frequent sips of oral rehydration solution (ORS).",
            "Seek urgent care if unable to keep fluids down or if bloody stools occur.",
        ),
    ),
    "Heart attack": DiseaseProfile(
        canonical_name="Heart attack",
        display_name="Myocardial Infarction (Heart Attack)",
        category="cardiovascular",
        base_urgency="Critical",
        specialist="Cardiologist / Emergency Medicine",
        contagious=False,
        actions=(
            "Call emergency services immediately — do not drive yourself to the hospital.",
            "Rest upright/seated while waiting for emergency responders.",
        ),
    ),
    "Hepatitis B": DiseaseProfile(
        canonical_name="Hepatitis B",
        display_name="Hepatitis B",
        category="hepatic_infectious",
        base_urgency="High",
        specialist="Hepatologist / Infectious Disease Specialist",
        contagious=True,
        actions=(
            "Seek serological viral panel and liver function testing promptly.",
            "Avoid alcohol and prevent blood/body-fluid exposure to others.",
        ),
    ),
    "Hepatitis C": DiseaseProfile(
        canonical_name="Hepatitis C",
        display_name="Hepatitis C",
        category="hepatic_infectious",
        base_urgency="High",
        specialist="Hepatologist / Infectious Disease Specialist",
        contagious=True,
        actions=(
            "Arrange confirmatory HCV RNA and liver function testing.",
            "Avoid alcohol and unprescribed medications while awaiting specialist care.",
        ),
    ),
    "Hepatitis D": DiseaseProfile(
        canonical_name="Hepatitis D",
        display_name="Hepatitis D",
        category="hepatic_infectious",
        base_urgency="High",
        specialist="Hepatologist / Infectious Disease Specialist",
        contagious=True,
        actions=(
            "Seek urgent hepatology evaluation along with Hepatitis B/D viral panel.",
            "Avoid alcohol and monitor closely for confusion or abdominal swelling.",
        ),
    ),
    "Hepatitis E": DiseaseProfile(
        canonical_name="Hepatitis E",
        display_name="Hepatitis E",
        category="hepatic_infectious",
        base_urgency="High",
        specialist="Hepatologist / Gastroenterologist",
        contagious=True,
        actions=(
            "Maintain strict hand and water hygiene and seek liver function testing.",
            "Pregnant patients with suspected Hepatitis E require immediate specialist care.",
        ),
    ),
    "Hypertension": DiseaseProfile(
        canonical_name="Hypertension",
        display_name="Hypertension",
        category="cardiovascular",
        base_urgency="Moderate",
        specialist="Cardiologist / General Physician",
        contagious=False,
        actions=(
            "Measure blood pressure calmly in a seated position.",
            "Seek emergency care if accompanied by chest pain, vision loss, or severe headache.",
        ),
    ),
    "Hyperthyroidism": DiseaseProfile(
        canonical_name="Hyperthyroidism",
        display_name="Hyperthyroidism",
        category="endocrine",
        base_urgency="Moderate",
        specialist="Endocrinologist",
        contagious=False,
        actions=(
            "Schedule thyroid panel (TSH, Free T3/T4) and endocrine consultation.",
            "Seek urgent care if rapid palpitations, high fever, or agitation develop.",
        ),
    ),
    "Hypoglycemia": DiseaseProfile(
        canonical_name="Hypoglycemia",
        display_name="Hypoglycemia",
        category="endocrine_acute",
        base_urgency="High",
        specialist="Endocrinologist / Emergency Medicine",
        contagious=False,
        actions=(
            "If alert and able to swallow safely, consume 15g fast-acting glucose and recheck blood sugar.",
            "Call emergency services if confused, drowsy, or unable to swallow.",
        ),
    ),
    "Hypothyroidism": DiseaseProfile(
        canonical_name="Hypothyroidism",
        display_name="Hypothyroidism",
        category="endocrine",
        base_urgency="Low",
        specialist="Endocrinologist",
        contagious=False,
        actions=(
            "Schedule TSH and Free T4 blood work with your clinician.",
            "Do not start or adjust thyroid hormone doses without medical supervision.",
        ),
    ),
    "Impetigo": DiseaseProfile(
        canonical_name="Impetigo",
        display_name="Impetigo",
        category="dermatological_infectious",
        base_urgency="Moderate",
        specialist="Dermatologist / General Physician",
        contagious=True,
        actions=(
            "Keep sores clean and covered; wash hands frequently to limit spread.",
            "Seek evaluation for prescribed topical or oral antibiotic therapy.",
        ),
    ),
    "Jaundice": DiseaseProfile(
        canonical_name="Jaundice",
        display_name="Jaundice (Hyperbilirubinemia)",
        category="hepatic",
        base_urgency="High",
        specialist="Gastroenterologist / Hepatologist",
        contagious=False,
        actions=(
            "Seek prompt evaluation for liver function tests and biliary ultrasound.",
            "Stay hydrated and avoid alcohol or unnecessary over-the-counter drugs.",
        ),
    ),
    "Malaria": DiseaseProfile(
        canonical_name="Malaria",
        display_name="Malaria",
        category="infectious_vector",
        base_urgency="High",
        specialist="Infectious Disease Specialist / Internal Medicine",
        contagious=False,
        actions=(
            "Obtain urgent blood smear or rapid diagnostic test (RDT) for malaria.",
            "Seek immediate medical treatment; avoid delaying antimalarial care.",
        ),
    ),
    "Migraine": DiseaseProfile(
        canonical_name="Migraine",
        display_name="Migraine",
        category="neurological",
        base_urgency="Moderate",
        specialist="Neurologist",
        contagious=False,
        actions=(
            "Rest in a quiet, dark room and maintain hydration.",
            "Seek emergency evaluation if this is a sudden worst-ever headache or has fever/stiff neck.",
        ),
    ),
    "Osteoarthristis": DiseaseProfile(
        canonical_name="Osteoarthristis",
        display_name="Osteoarthritis",
        category="musculoskeletal",
        base_urgency="Low",
        specialist="Orthopedist",
        contagious=False,
        actions=(
            "Use gentle low-impact mobility exercises and joint protection.",
            "Consult an orthopedist or physiotherapist for a long-term management plan.",
        ),
    ),
    "Paralysis (brain hemorrhage)": DiseaseProfile(
        canonical_name="Paralysis (brain hemorrhage)",
        display_name="Acute Stroke / Brain Hemorrhage",
        category="neurological_critical",
        base_urgency="Critical",
        specialist="Neurologist / Emergency Medicine",
        contagious=False,
        actions=(
            "Call emergency services immediately — stroke care is strictly time-critical.",
            "Note the exact time symptoms started and keep the airway clear.",
        ),
    ),
    "Peptic ulcer diseae": DiseaseProfile(
        canonical_name="Peptic ulcer diseae",
        display_name="Peptic Ulcer Disease",
        category="gastrointestinal",
        base_urgency="Moderate",
        specialist="Gastroenterologist",
        contagious=False,
        actions=(
            "Avoid NSAID painkillers, alcohol, and smoking; seek H. pylori/GI assessment.",
            "Go to emergency care immediately if black stools or vomiting blood occur.",
        ),
    ),
    "Pneumonia": DiseaseProfile(
        canonical_name="Pneumonia",
        display_name="Pneumonia",
        category="respiratory_infectious",
        base_urgency="High",
        specialist="Pulmonologist / Internal Medicine",
        contagious=True,
        actions=(
            "Seek prompt chest examination, pulse oximetry, and chest X-ray.",
            "Go to emergency care immediately if breathing becomes labored or oxygen drops.",
        ),
    ),
    "Psoriasis": DiseaseProfile(
        canonical_name="Psoriasis",
        display_name="Psoriasis",
        category="dermatological",
        base_urgency="Low",
        specialist="Dermatologist",
        contagious=False,
        actions=(
            "Moisturize skin regularly and avoid harsh skin trauma or scrubbing.",
            "Consult a dermatologist for topical or systemic disease-modifying options.",
        ),
    ),
    "Tuberculosis": DiseaseProfile(
        canonical_name="Tuberculosis",
        display_name="Tuberculosis (TB)",
        category="respiratory_infectious",
        base_urgency="High",
        specialist="Pulmonologist / Infectious Disease Specialist",
        contagious=True,
        actions=(
            "Wear a well-fitting mask, isolate in a ventilated space, and seek sputum/chest X-ray testing.",
            "Begin supervised medical evaluation promptly — TB requires full multi-drug regimens.",
        ),
    ),
    "Typhoid": DiseaseProfile(
        canonical_name="Typhoid",
        display_name="Typhoid Fever",
        category="infectious_bacterial",
        base_urgency="High",
        specialist="Infectious Disease Specialist / General Physician",
        contagious=True,
        actions=(
            "Seek prompt physician evaluation and blood culture testing before antibiotics.",
            "Maintain hydration and strict hand/food hygiene to prevent transmission.",
        ),
    ),
    "Urinary tract infection": DiseaseProfile(
        canonical_name="Urinary tract infection",
        display_name="Urinary Tract Infection (UTI)",
        category="urological_infectious",
        base_urgency="Moderate",
        specialist="General Physician / Urologist",
        contagious=False,
        actions=(
            "Increase fluid intake and arrange urinalysis / urine culture.",
            "Seek urgent care if high fever, flank/back pain, or vomiting develops.",
        ),
    ),
    "Varicose veins": DiseaseProfile(
        canonical_name="Varicose veins",
        display_name="Varicose Veins",
        category="vascular",
        base_urgency="Low",
        specialist="Vascular Surgeon",
        contagious=False,
        actions=(
            "Elevate legs when resting and avoid prolonged motionless standing.",
            "Seek urgent care if a leg becomes acutely swollen, hot, or painful.",
        ),
    ),
    "hepatitis A": DiseaseProfile(
        canonical_name="hepatitis A",
        display_name="Hepatitis A",
        category="hepatic_infectious",
        base_urgency="Moderate",
        specialist="Gastroenterologist / Hepatologist",
        contagious=True,
        actions=(
            "Practice strict hand hygiene and avoid preparing food for others.",
            "Rest, hydrate, and avoid alcohol/acetaminophen until liver tests recover.",
        ),
    ),
}


def get_disease_profile(disease_name: str) -> DiseaseProfile:
    """Return ontology record for a disease, with safe fallback for unknown names."""
    if disease_name in DISEASE_KB:
        return DISEASE_KB[disease_name]
    return DiseaseProfile(
        canonical_name=disease_name,
        display_name=disease_name,
        category="general_medical",
        base_urgency="Moderate",
        specialist="General Physician",
        contagious=False,
        actions=("Consult a licensed healthcare professional for clinical evaluation.",),
    )
