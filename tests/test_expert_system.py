"""Rule-validation and ML-integration tests for the Forward-Chaining Expert System."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.logic.expert_system import TriageExpertSystem
from backend.logic.knowledge_base import DISEASE_KB
from backend.ml.naive_bayes import PredictionAnalysis, _BaseNaiveBayes

ARTIFACT_DIR = Path("backend/artifacts")


def _make_analysis(
    top_predictions: list[tuple[str, float]],
    entropy: float = 0.25,
    margin: float | None = None,
) -> PredictionAnalysis:
    preds = [
        {"disease": name, "probability": prob, "rank": i + 1}
        for i, (name, prob) in enumerate(top_predictions)
    ]
    if margin is None:
        margin = (
            top_predictions[0][1] - top_predictions[1][1]
            if len(top_predictions) > 1
            else 1.0
        )
    return PredictionAnalysis(
        top_predictions=preds,
        normalized_entropy=entropy,
        confidence_margin=margin,
        symptom_contributions=[],
        discriminating_followups=[],
    )


def test_knowledge_base_covers_all_41_trained_diseases() -> None:
    meta_path = ARTIFACT_DIR / "nb_model_meta.json"
    assert meta_path.exists(), "Trained model metadata missing."
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = set(meta["classes"]) - set(DISEASE_KB.keys())
    assert not missing, f"Missing diseases in DISEASE_KB: {missing}"


def test_cardiac_red_flag_overrides_benign_ml_prediction() -> None:
    es = TriageExpertSystem()
    analysis = _make_analysis([("GERD", 0.55), ("Heart attack", 0.25)])
    triage = es.evaluate(["chest_pain", "sweating"], analysis)

    assert triage.urgency == "Critical"
    assert "Cardiologist" in triage.specialist
    assert any("Cardiac" in rf for rf in triage.red_flags)
    fired_rules = {step.rule_name for step in triage.inference_trace}
    assert "R2_CardiacRedFlag" in fired_rules
    assert "R14_CriticalEmergencyAction" in fired_rules


def test_stroke_and_meningeal_red_flags() -> None:
    es = TriageExpertSystem()
    stroke_res = es.evaluate(
        ["weakness_of_one_body_side", "slurred_speech"],
        _make_analysis([("Migraine", 0.60), ("Paralysis (brain hemorrhage)", 0.30)]),
    )
    assert stroke_res.urgency == "Critical"
    assert any("Stroke" in rf for rf in stroke_res.red_flags)

    meningeal_res = es.evaluate(
        ["high_fever", "stiff_neck", "headache"],
        _make_analysis([("Typhoid", 0.70), ("Malaria", 0.15)]),
    )
    assert meningeal_res.urgency == "Critical"
    assert any("Meningeal" in rf for rf in meningeal_res.red_flags)


def test_multi_hop_differential_safety_escalates_without_red_flag_symptoms() -> None:
    es = TriageExpertSystem()
    # Low confidence GERD vs plausible Heart attack triggers R8 -> R10 -> R12 -> R14
    analysis = _make_analysis([("GERD", 0.42), ("Heart attack", 0.31)], margin=0.11)
    triage = es.evaluate(["vomiting", "indigestion"], analysis)

    assert triage.differential_alert is True
    assert triage.urgency == "Critical"
    fired_rules = [step.rule_name for step in triage.inference_trace]
    assert "R8_LowConfidenceDifferential" in fired_rules
    assert "R10_HighUrgencyDifferentialSafety" in fired_rules
    assert "R14_CriticalEmergencyAction" in fired_rules


def test_contagious_precaution_and_sparse_input_warning() -> None:
    es = TriageExpertSystem()
    tb_res = es.evaluate(
        ["cough"],
        _make_analysis([("Tuberculosis", 0.92), ("Common Cold", 0.05)]),
    )
    assert tb_res.differential_alert is True
    assert any("1 symptom" in w for w in tb_res.warnings)
    assert any("communicable" in p for p in tb_res.precautions)


def test_end_to_end_real_trained_model_and_expert_system() -> None:
    model = _BaseNaiveBayes.load(ARTIFACT_DIR, prefix="nb_model")
    es = TriageExpertSystem()

    x = np.zeros(len(model.feature_names_), dtype=np.int64)
    symptoms = ["yellowish_skin", "dark_urine", "abdominal_pain"]
    for s in symptoms:
        x[model.feature_names_.index(s)] = 1

    analysis = model.predict_detailed(x, top_k=3)
    triage = es.evaluate(symptoms, analysis)
    assert analysis.top_predictions[0]["disease"] in ("Jaundice", "Hepatitis D", "Chronic cholestasis")
    assert triage.differential_alert is True
    assert triage.urgency in ("High", "Critical")
    assert len(analysis.discriminating_followups) > 0
    assert len(triage.inference_trace) >= 3
