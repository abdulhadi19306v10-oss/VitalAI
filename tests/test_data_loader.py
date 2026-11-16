"""Unit tests for CSV cleaning, schema validation, deduplication, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.ml.data_loader import (
    DatasetValidationError,
    deduplicate_patterns,
    format_symptom_display,
    load_dataset,
    normalize_symptom_key,
    stratified_kfold_indices,
)
from backend.ml.symptom_aliases import (
    CANONICAL_SYNONYMS,
    DISPLAY_NAME_OVERRIDES,
    SYMPTOM_ALIASES,
    resolve_symptom_key,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "mini_disease_dataset.csv"


def test_load_fixture_strips_unnamed_and_normalizes_labels() -> None:
    bundle = load_dataset(FIXTURE_CSV)
    assert bundle.X_train.shape == (60, 15)
    assert len(bundle.class_names) == 6
    assert "high_fever" in bundle.feature_names
    assert "Heart attack" in bundle.class_names
    assert "Fungal infection" in bundle.class_names
    assert bundle.diagnostics["train_dedup"]["duplicate_rows"] == 12
    assert bundle.diagnostics["train_dedup"]["conflicting_vectors"] == 0


def test_normalize_symptom_key_handles_kaggle_quirks() -> None:
    assert normalize_symptom_key("spotting_ urination") == "spotting_urination"
    assert normalize_symptom_key("dischromic _patches") == "dischromic_patches"
    assert normalize_symptom_key("foul_smell_of urine") == "foul_smell_of_urine"


def test_deduplicate_and_stratified_kfold() -> None:
    bundle = load_dataset(FIXTURE_CSV)
    X_dedup, y_dedup, stats = deduplicate_patterns(bundle.X_train, bundle.y_train)
    assert stats["unique_pairs"] == 48
    assert X_dedup.shape[0] == 48

    splits = stratified_kfold_indices(y_dedup, n_splits=4, seed=42)
    assert len(splits) == 4
    seen_val: list[int] = []
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
        seen_val.extend(val_idx.tolist())
    assert sorted(seen_val) == list(range(48))


def test_invalid_non_binary_csv_rejected(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("fever,cough,prognosis\n2,0,Flu\n0,1,Cold\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="only 0/1"):
        load_dataset(bad_csv)


def test_mismatched_test_columns_rejected(tmp_path: Path) -> None:
    test_csv = tmp_path / "test_bad.csv"
    test_csv.write_text("fever,cough,prognosis\n1,0,Heart attack\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Feature mismatch"):
        load_dataset(FIXTURE_CSV, test_csv)


def test_merges_pandas_duplicate_suffix_columns(tmp_path: Path) -> None:
    dup_csv = tmp_path / "dup_col.csv"
    dup_csv.write_text(
        "fluid_overload,cough,fluid_overload.1,prognosis\n"
        "0,0,1,Alcoholic hepatitis\n"
        "0,1,0,Pneumonia\n",
        encoding="utf-8",
    )
    bundle = load_dataset(dup_csv)
    assert bundle.feature_names == ["fluid_overload", "cough"]
    assert bundle.X_train.shape == (2, 2)
    assert bundle.X_train[0, 0] == 1
    assert bundle.X_train[1, 0] == 0


def test_symptom_aliases_are_valid_and_conflict_free() -> None:
    import json

    meta_path = Path("backend/artifacts/nb_model_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = set(meta["feature_names"])

    assert set(CANONICAL_SYNONYMS.keys()).issubset(features)
    assert set(DISPLAY_NAME_OVERRIDES.keys()).issubset(features)
    collisions = set(SYMPTOM_ALIASES.keys()) & features
    assert not collisions, f"Alias collided with canonical feature(s): {collisions}"

    assert resolve_symptom_key("Shortness of Breath") == "breathlessness"
    assert resolve_symptom_key("dyspnea") == "breathlessness"
    assert resolve_symptom_key("diarrhea") == "diarrhoea"
    assert resolve_symptom_key("swollen extremities") == "swollen_extremeties"
    assert format_symptom_display("swollen_extremeties") == "Swollen Extremities"


