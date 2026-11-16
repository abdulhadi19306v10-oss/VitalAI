"""Dataset loading, schema validation, deduplication, and stratified splitting."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.symptom_aliases import DISPLAY_NAME_OVERRIDES, normalize_token


class DatasetValidationError(ValueError):
    """Raised when an input CSV fails schema, alignment, or value checks."""


@dataclass
class DatasetBundle:
    """Validated training and optional testing arrays with canonical metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    feature_names: list[str]
    feature_display_names: dict[str, str]
    class_names: list[str]
    diagnostics: dict[str, Any]


def normalize_symptom_key(raw: str) -> str:
    """Convert raw symptom column names to clean canonical snake_case keys."""
    return normalize_token(raw)


def format_symptom_display(key: str) -> str:
    """Convert a canonical symptom key into a human-readable label."""
    if key in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[key]
    return " ".join(part.capitalize() for part in key.split("_") if part)


def normalize_disease_name(raw: str) -> str:
    """Normalize whitespace in disease target labels."""
    return re.sub(r"\s+", " ", str(raw).strip())


def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing empty Unnamed columns produced by stray CSV commas."""
    to_drop: list[str] = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.startswith("Unnamed:") or col_str == "":
            series = df[col]
            is_all_empty = series.isna().all() or (
                series.astype(str).str.strip().eq("").all()
            )
            if not is_all_empty:
                raise DatasetValidationError(
                    f"Unexpected non-empty unnamed column '{col}' found in CSV."
                )
            to_drop.append(col)
    return df.drop(columns=to_drop) if to_drop else df


def _load_and_clean_csv(
    path: Path, target_col: str = "prognosis"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a symptom CSV, validate binary features, and normalize labels."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise DatasetValidationError(f"Dataset file is empty: {path}")

    df = _drop_empty_unnamed_columns(df)
    df.columns = [str(c).strip() for c in df.columns]

    target_matches = [c for c in df.columns if c.lower() == target_col.lower()]
    if not target_matches:
        raise DatasetValidationError(
            f"Target column '{target_col}' missing in {path.name}."
        )
    if len(target_matches) > 1:
        raise DatasetValidationError(
            f"Duplicate target columns matching '{target_col}' in {path.name}."
        )
    actual_target = target_matches[0]

    raw_features = [c for c in df.columns if c != actual_target]
    if not raw_features:
        raise DatasetValidationError(f"No symptom feature columns found in {path.name}.")

    norm_features = [normalize_symptom_key(c) for c in raw_features]
    dup_features = [name for name, count in Counter(norm_features).items() if count > 1]
    if dup_features:
        raise DatasetValidationError(
            f"Duplicate normalized feature names in {path.name}: {dup_features}"
        )

    feature_df = df[raw_features].copy()
    if feature_df.isna().any().any():
        bad_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise DatasetValidationError(
            f"Missing values detected in feature columns of {path.name}: {bad_cols[:5]}"
        )

    try:
        numeric_df = feature_df.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError(
            f"Non-numeric feature values in {path.name}: {exc}"
        ) from exc

    # Merge pandas-renamed duplicate columns (e.g. 'fluid_overload' and 'fluid_overload.1')
    raw_to_norm = {raw: norm for raw, norm in zip(raw_features, norm_features)}
    norm_to_raw = {norm: raw for raw, norm in raw_to_norm.items()}
    cols_to_drop: list[str] = []
    for raw_col in raw_features:
        match = re.match(r"^(?P<base>.+)\.(?P<idx>\d+)$", raw_col)
        if match:
            base_norm = normalize_symptom_key(match.group("base"))
            if base_norm in norm_to_raw:
                base_raw = norm_to_raw[base_norm]
                numeric_df[base_raw] = np.maximum(
                    numeric_df[base_raw].to_numpy(dtype=np.float64),
                    numeric_df[raw_col].to_numpy(dtype=np.float64),
                )
                cols_to_drop.append(raw_col)

    if cols_to_drop:
        drop_set = set(cols_to_drop)
        kept_pairs = [
            (raw, norm)
            for raw, norm in zip(raw_features, norm_features)
            if raw not in drop_set
        ]
        raw_features = [raw for raw, _ in kept_pairs]
        norm_features = [norm for _, norm in kept_pairs]
        numeric_df = numeric_df[raw_features]

    X = numeric_df.to_numpy(dtype=np.float64)
    if not np.isin(X, [0.0, 1.0]).all():
        invalid = np.unique(X[~np.isin(X, [0.0, 1.0])])
        raise DatasetValidationError(
            f"Feature matrix in {path.name} must contain only 0/1 values; found {invalid[:5]}"
        )

    target_series = df[actual_target]
    if target_series.isna().any():
        raise DatasetValidationError(f"Missing target labels in {path.name}.")

    y = np.array(
        [normalize_disease_name(val) for val in target_series.tolist()], dtype=object
    )
    if any(label == "" for label in y):
        raise DatasetValidationError(f"Empty string target label found in {path.name}.")

    return X.astype(np.int64), y, norm_features


def deduplicate_patterns(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collapse exact duplicate (symptom_vector, label) pairs and report conflicts."""
    seen: dict[tuple[bytes, str], int] = {}
    unique_indices: list[int] = []
    pattern_to_labels: dict[bytes, set[str]] = defaultdict(set)

    for idx in range(X.shape[0]):
        row_bytes = np.ascontiguousarray(X[idx], dtype=np.int64).tobytes()
        label = str(y[idx])
        pattern_to_labels[row_bytes].add(label)
        key = (row_bytes, label)
        if key not in seen:
            seen[key] = idx
            unique_indices.append(idx)

    conflicting = sum(1 for labels in pattern_to_labels.values() if len(labels) > 1)
    idx_arr = np.array(unique_indices, dtype=np.int64)
    stats = {
        "total_rows": int(X.shape[0]),
        "unique_pairs": int(len(unique_indices)),
        "duplicate_rows": int(X.shape[0] - len(unique_indices)),
        "duplicate_ratio": float((X.shape[0] - len(unique_indices)) / max(1, X.shape[0])),
        "unique_symptom_vectors": int(len(pattern_to_labels)),
        "conflicting_vectors": int(conflicting),
    }
    return X[idx_arr], y[idx_arr], stats


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified K-fold splitter built with NumPy."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(y) < n_splits:
        raise ValueError(
            f"Cannot split {len(y)} samples into {n_splits} folds."
        )

    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    unique_classes = sorted(set(str(label) for label in y))
    for class_offset, cls in enumerate(unique_classes):
        cls_idx = np.where(y == cls)[0]
        shuffled = rng.permutation(cls_idx)
        for pos, sample_idx in enumerate(shuffled):
            fold_id = (pos + class_offset) % n_splits
            folds[fold_id].append(int(sample_idx))

    all_indices = np.arange(len(y), dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        val_idx = np.sort(np.array(folds[fold_id], dtype=np.int64))
        if val_idx.size == 0:
            continue
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, val_idx))

    if len(splits) < 2:
        raise ValueError("Dataset too small to produce at least 2 non-empty folds.")
    return splits


def load_dataset(
    train_path: Path,
    test_path: Path | None = None,
    target_col: str = "prognosis",
) -> DatasetBundle:
    """Load, validate, and align Training.csv and optional Testing.csv."""
    X_train, y_train, train_features = _load_and_clean_csv(train_path, target_col)

    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if test_path is not None and test_path.exists():
        X_test_raw, y_test, test_features = _load_and_clean_csv(test_path, target_col)
        if set(test_features) != set(train_features):
            missing = sorted(set(train_features) - set(test_features))
            extra = sorted(set(test_features) - set(train_features))
            raise DatasetValidationError(
                f"Feature mismatch between train and test CSVs. "
                f"Missing in test: {missing[:5]}; Extra in test: {extra[:5]}"
            )
        test_col_index = {name: i for i, name in enumerate(test_features)}
        reorder = [test_col_index[name] for name in train_features]
        X_test = X_test_raw[:, reorder]

        unseen_test_classes = sorted(set(y_test.tolist()) - set(y_train.tolist()))
        if unseen_test_classes:
            raise DatasetValidationError(
                f"Test set contains classes absent from training set: {unseen_test_classes}"
            )

    _, _, train_dedup_stats = deduplicate_patterns(X_train, y_train)
    class_counts = dict(sorted(Counter(y_train.tolist()).items()))
    class_names = sorted(class_counts.keys())

    zero_symptom_train = int((X_train.sum(axis=1) == 0).sum())
    diagnostics: dict[str, Any] = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]) if X_test is not None else 0,
        "n_features": int(len(train_features)),
        "n_classes": int(len(class_names)),
        "zero_symptom_train_rows": zero_symptom_train,
        "train_dedup": train_dedup_stats,
        "min_class_count": min(class_counts.values()),
        "max_class_count": max(class_counts.values()),
        "class_counts": class_counts,
    }
    if X_test is not None and y_test is not None:
        _, _, test_dedup_stats = deduplicate_patterns(X_test, y_test)
        diagnostics["test_dedup"] = test_dedup_stats

    display_map = {name: format_symptom_display(name) for name in train_features}
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=train_features,
        feature_display_names=display_map,
        class_names=class_names,
        diagnostics=diagnostics,
    )
