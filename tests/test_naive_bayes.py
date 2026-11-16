"""Tests for self-written Naive Bayes, KNN baseline, evaluation, and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.ml.data_loader import load_dataset
from backend.ml.evaluate_model import run_full_evaluation
from backend.ml.knn_baseline import CosineKNNClassifier
from backend.ml.naive_bayes import (
    BernoulliNaiveBayes,
    MultinomialNaiveBayes,
    _BaseNaiveBayes,
)
from backend.ml.train_model import train_and_save

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "mini_disease_dataset.csv"


def test_multinomial_and_bernoulli_probabilities_and_roundtrip(tmp_path: Path) -> None:
    bundle = load_dataset(FIXTURE_CSV)
    for cls, prefix in ((MultinomialNaiveBayes, "mnb"), (BernoulliNaiveBayes, "bnb")):
        model = cls(alpha=0.5).fit(
            bundle.X_train, bundle.y_train, feature_names=bundle.feature_names
        )
        probs = model.predict_proba(bundle.X_train)
        assert probs.shape == (bundle.X_train.shape[0], len(bundle.class_names))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        saved_path = model.save(tmp_path, prefix=prefix)
        assert saved_path.exists()
        loaded = _BaseNaiveBayes.load(tmp_path, prefix=prefix)
        loaded_probs = loaded.predict_proba(bundle.X_train)
        assert np.allclose(probs, loaded_probs, atol=1e-10)


def test_predict_detailed_rejects_zero_symptoms_and_explains() -> None:
    bundle = load_dataset(FIXTURE_CSV)
    model = MultinomialNaiveBayes(alpha=0.5).fit(
        bundle.X_train, bundle.y_train, feature_names=bundle.feature_names
    )
    with pytest.raises(ValueError, match="At least one active symptom"):
        model.predict_detailed(np.zeros(len(bundle.feature_names)))

    x = np.zeros(len(bundle.feature_names), dtype=np.int64)
    for sym in ("chest_pain", "breathlessness", "sweating"):
        x[bundle.feature_names.index(sym)] = 1

    analysis = model.predict_detailed(x, top_k=3)
    assert analysis.top_predictions[0]["disease"] == "Heart attack"
    assert len(analysis.top_predictions) == 3
    assert 0.0 <= analysis.normalized_entropy <= 1.0
    assert len(analysis.symptom_contributions) == 3
    assert len(analysis.discriminating_followups) > 0


def test_knn_baseline_runs_cleanly() -> None:
    bundle = load_dataset(FIXTURE_CSV)
    knn = CosineKNNClassifier(k=3, deduplicate=True).fit(bundle.X_train, bundle.y_train)
    preds = knn.predict(bundle.X_train)
    acc = float(np.mean(preds == bundle.y_train))
    assert acc >= 0.95


def test_full_evaluation_and_train_pipeline(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    summary = train_and_save(
        train_path=FIXTURE_CSV,
        test_path=None,
        artifact_dir=artifact_dir,
        model_type="auto",
        n_splits=4,
        seed=42,
    )
    assert (artifact_dir / "nb_model.npz").exists()
    assert (artifact_dir / "nb_model_meta.json").exists()
    assert (artifact_dir / "evaluation_report.json").exists()
    assert (artifact_dir / "alpha_tuning_curve.png").exists()
    assert (artifact_dir / "confusion_matrix.png").exists()
    assert (artifact_dir / "model_comparison.png").exists()

    audit = summary["overfitting_audit"]
    assert audit["cv_error_rate"] <= 0.05
    assert abs(audit["cv_generalization_gap"]) <= 0.05
    assert audit["status"] == "PASS_LOW_MISTAKE_NO_OVERFIT"
