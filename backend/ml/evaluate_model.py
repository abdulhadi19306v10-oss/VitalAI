"""Comprehensive ML evaluation, overfitting audit, alpha tuning, and plots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.ml.data_loader import (
    DatasetBundle,
    deduplicate_patterns,
    load_dataset,
    stratified_kfold_indices,
)
from backend.ml.knn_baseline import CosineKNNClassifier
from backend.ml.naive_bayes import (
    BernoulliNaiveBayes,
    MultinomialNaiveBayes,
    _BaseNaiveBayes,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute accuracy, error rate, macro/weighted P/R/F1, log-loss, and confusion matrix."""
    n_classes = len(classes)
    cls_to_idx = {cls: i for i, cls in enumerate(classes)}
    true_idx = np.array([cls_to_idx[str(label)] for label in y_true], dtype=np.int64)
    pred_idx = np.array([cls_to_idx[str(label)] for label in y_pred], dtype=np.int64)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true_idx, pred_idx), 1)

    total = max(1, len(y_true))
    correct = int(np.trace(cm))
    accuracy = float(correct / total)
    error_rate = float(1.0 - accuracy)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    precision_per = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_denom = precision_per + recall_per
    f1_per = np.divide(
        2.0 * precision_per * recall_per,
        f1_denom,
        out=np.zeros_like(tp),
        where=f1_denom > 0,
    )

    present_mask = support > 0
    macro_precision = float(np.mean(precision_per[present_mask])) if present_mask.any() else 0.0
    macro_recall = float(np.mean(recall_per[present_mask])) if present_mask.any() else 0.0
    macro_f1 = float(np.mean(f1_per[present_mask])) if present_mask.any() else 0.0
    weighted_f1 = float(np.sum(f1_per * support) / max(1.0, support.sum()))

    log_loss: float | None = None
    top3_accuracy: float | None = None
    if y_proba is not None and y_proba.shape == (len(y_true), n_classes):
        clipped = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
        true_probs = clipped[np.arange(len(y_true)), true_idx]
        log_loss = float(-np.mean(np.log(true_probs)))
        k_top = min(3, n_classes)
        top_k_indices = np.argsort(y_proba, axis=1)[:, -k_top:]
        top3_hits = sum(
            1 for row_i, t_idx in enumerate(true_idx) if t_idx in top_k_indices[row_i]
        )
        top3_accuracy = float(top3_hits / total)

    confusions: list[dict[str, Any]] = []
    for t_i in range(n_classes):
        for p_i in range(n_classes):
            if t_i != p_i and cm[t_i, p_i] > 0:
                confusions.append(
                    {
                        "true": classes[t_i],
                        "predicted": classes[p_i],
                        "count": int(cm[t_i, p_i]),
                    }
                )
    confusions.sort(key=lambda item: item["count"], reverse=True)

    per_class = [
        {
            "disease": classes[i],
            "support": int(support[i]),
            "precision": round(float(precision_per[i]), 4),
            "recall": round(float(recall_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "errors": int(fn[i]),
        }
        for i in range(n_classes)
        if support[i] > 0
    ]

    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(accuracy, 6),
        "error_rate": round(error_rate, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "log_loss": round(log_loss, 6) if log_loss is not None else None,
        "top3_accuracy": round(top3_accuracy, 6) if top3_accuracy is not None else None,
        "top_confusions": confusions[:10],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def perturb_symptoms(
    X: np.ndarray,
    drop_rate: float = 0.2,
    noise_rate: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate incomplete patient reporting by dropping active symptoms (min 1 kept)."""
    rng = np.random.default_rng(seed)
    X_pert = X.copy()
    for i in range(X_pert.shape[0]):
        active = np.where(X_pert[i] > 0)[0]
        if len(active) > 1 and drop_rate > 0:
            drop_mask = rng.random(len(active)) < drop_rate
            if drop_mask.all():
                keep_idx = int(rng.integers(0, len(active)))
                drop_mask[keep_idx] = False
            X_pert[i, active[drop_mask]] = 0
        if noise_rate > 0:
            inactive = np.where(X_pert[i] == 0)[0]
            if len(inactive) > 0:
                add_mask = rng.random(len(inactive)) < noise_rate
                X_pert[i, inactive[add_mask]] = 1
    return X_pert


def _align_proba_to_classes(
    model_classes: np.ndarray, proba: np.ndarray, target_classes: list[str]
) -> np.ndarray:
    """Align model class probability columns when a CV fold misses a rare class."""
    if list(model_classes) == target_classes:
        return proba
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float64)
    target_idx = {cls: i for i, cls in enumerate(target_classes)}
    for m_i, cls in enumerate(model_classes):
        if str(cls) in target_idx:
            aligned[:, target_idx[str(cls)]] = proba[:, m_i]
    return aligned


def run_cross_validation(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    n_splits: int = 5,
    deduplicate: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run stratified K-fold CV (optionally deduplicated) and measure generalization gap."""
    X_cv, y_cv = X, y
    cv_mode = "full_stratified"
    if deduplicate:
        X_dedup, y_dedup, _ = deduplicate_patterns(X, y)
        min_per_class = min(
            int(np.sum(y_dedup == cls)) for cls in sorted(set(y_dedup.tolist()))
        )
        if len(X_dedup) >= n_splits and min_per_class >= 2:
            X_cv, y_cv = X_dedup, y_dedup
            cv_mode = "deduplicated_stratified"
        else:
            cv_mode = "full_stratified_fallback"

    effective_splits = min(n_splits, len(X_cv))
    splits = stratified_kfold_indices(y_cv, n_splits=effective_splits, seed=seed)

    train_accs: list[float] = []
    val_accs: list[float] = []
    noisy_val_accs: list[float] = []
    val_losses: list[float] = []
    oof_preds = np.empty(len(y_cv), dtype=object)
    oof_proba = np.zeros((len(y_cv), len(classes)), dtype=np.float64)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        model = model_factory()
        model.fit(X_cv[train_idx], y_cv[train_idx])

        tr_pred = model.predict(X_cv[train_idx])
        train_accs.append(float(np.mean(tr_pred == y_cv[train_idx])))

        va_pred = model.predict(X_cv[val_idx])
        va_proba = _align_proba_to_classes(
            model.classes_, model.predict_proba(X_cv[val_idx]), classes
        )
        oof_preds[val_idx] = va_pred
        oof_proba[val_idx] = va_proba

        X_val_noisy = perturb_symptoms(
            X_cv[val_idx], drop_rate=0.20, noise_rate=0.02, seed=seed + fold_i
        )
        noisy_val_accs.append(float(np.mean(model.predict(X_val_noisy) == y_cv[val_idx])))

        val_metrics = compute_classification_metrics(
            y_cv[val_idx], va_pred, classes, va_proba
        )
        val_accs.append(float(val_metrics["accuracy"]))
        if val_metrics["log_loss"] is not None:
            val_losses.append(float(val_metrics["log_loss"]))

    oof_metrics = compute_classification_metrics(y_cv, oof_preds, classes, oof_proba)
    mean_train_acc = float(np.mean(train_accs))
    mean_val_acc = float(np.mean(val_accs))
    mean_noisy_val_acc = float(np.mean(noisy_val_accs))
    return {
        "cv_mode": cv_mode,
        "n_splits": len(splits),
        "n_cv_samples": int(len(y_cv)),
        "mean_train_accuracy": round(mean_train_acc, 6),
        "mean_val_accuracy": round(mean_val_acc, 6),
        "std_val_accuracy": round(float(np.std(val_accs)), 6),
        "mean_val_error_rate": round(1.0 - mean_val_acc, 6),
        "mean_noisy_val_accuracy": round(mean_noisy_val_acc, 6),
        "mean_noisy_val_error_rate": round(1.0 - mean_noisy_val_acc, 6),
        "mean_val_log_loss": round(float(np.mean(val_losses)), 6) if val_losses else None,
        "generalization_gap": round(mean_train_acc - mean_val_acc, 6),
        "oof_metrics": oof_metrics,
    }


def tune_alpha(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    classes: list[str],
    model_cls: type[_BaseNaiveBayes] = MultinomialNaiveBayes,
    alphas: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    n_splits: int = 5,
    fit_prior: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep Laplace smoothing alpha and pick the lowest-error, well-regularized value."""
    sweep_results: list[dict[str, Any]] = []
    for alpha in alphas:
        cv_res = run_cross_validation(
            model_factory=lambda a=alpha: model_cls(alpha=a, fit_prior=fit_prior),
            X=X,
            y=y,
            classes=classes,
            n_splits=n_splits,
            deduplicate=True,
            seed=seed,
        )

        sweep_results.append(
            {
                "alpha": float(alpha),
                "mean_train_accuracy": cv_res["mean_train_accuracy"],
                "cv_val_accuracy": cv_res["mean_val_accuracy"],
                "cv_error_rate": cv_res["mean_val_error_rate"],
                "noisy_cv_val_accuracy": cv_res["mean_noisy_val_accuracy"],
                "noisy_cv_error_rate": cv_res["mean_noisy_val_error_rate"],
                "cv_log_loss": cv_res["mean_val_log_loss"],
                "generalization_gap": cv_res["generalization_gap"],
            }
        )

    # Optimize out-of-fold clean error, then noisy/dropout robustness, then log-loss, then stronger alpha
    best = min(
        sweep_results,
        key=lambda row: (
            round(row["cv_error_rate"], 4),
            round(row["noisy_cv_error_rate"], 4),
            round(row["cv_log_loss"] if row["cv_log_loss"] is not None else 999.0, 4),
            -row["alpha"],
        ),
    )
    return {
        "model_type": model_cls.model_type,
        "best_alpha": best["alpha"],
        "best_summary": best,
        "sweep": sweep_results,
    }


def assess_overfitting_and_quality(
    train_acc: float,
    cv_summary: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    dropout_20_acc: float,
) -> dict[str, Any]:
    """Evaluate explicit overfitting and mistake-rate thresholds."""
    cv_gap = float(cv_summary["generalization_gap"])
    cv_err = float(cv_summary["mean_val_error_rate"])
    test_gap = (
        float(train_acc - test_metrics["accuracy"]) if test_metrics is not None else None
    )
    test_err = float(test_metrics["error_rate"]) if test_metrics is not None else None
    dropout_err = float(1.0 - dropout_20_acc)

    max_gap = max(cv_gap, test_gap if test_gap is not None else 0.0)
    worst_clean_err = max(cv_err, test_err if test_err is not None else 0.0)

    reasons: list[str] = []
    if max_gap > 0.08:
        reasons.append(f"High train-vs-validation/test generalization gap ({max_gap:.3f}).")
    elif max_gap > 0.04:
        reasons.append(f"Moderate generalization gap ({max_gap:.3f}).")

    if worst_clean_err > 0.08:
        reasons.append(f"High clean mistake rate ({worst_clean_err:.3f}).")
    elif worst_clean_err > 0.04:
        reasons.append(f"Moderate clean mistake rate ({worst_clean_err:.3f}).")

    if dropout_err > 0.15:
        reasons.append(f"Elevated 20%-symptom-dropout mistake rate ({dropout_err:.3f}).")

    if max_gap <= 0.04 and worst_clean_err <= 0.04 and dropout_err <= 0.15:
        status = "PASS_LOW_MISTAKE_NO_OVERFIT"
    elif max_gap <= 0.08 and worst_clean_err <= 0.08:
        status = "WARN_MODERATE_GAP_OR_ERROR"
    else:
        status = "FAIL_OVERFIT_OR_HIGH_ERROR"

    return {
        "status": status,
        "train_accuracy": round(train_acc, 6),
        "cv_generalization_gap": round(cv_gap, 6),
        "test_generalization_gap": round(test_gap, 6) if test_gap is not None else None,
        "cv_error_rate": round(cv_err, 6),
        "test_error_rate": round(test_err, 6) if test_err is not None else None,
        "dropout_20_error_rate": round(dropout_err, 6),
        "notes": reasons if reasons else ["Generalization gap and mistake rate are within strict thresholds."],
    }


def _evaluate_single_model(
    name: str,
    model_factory: Callable[[], Any],
    bundle: DatasetBundle,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    """Fit a model, measure train/test/CV/dropout metrics, and return evaluation record."""
    t0 = time.perf_counter()
    model = model_factory()
    if hasattr(model, "feature_names_"):
        model.fit(bundle.X_train, bundle.y_train, feature_names=bundle.feature_names)
    else:
        model.fit(bundle.X_train, bundle.y_train)
    train_time_ms = (time.perf_counter() - t0) * 1000.0

    train_pred = model.predict(bundle.X_train)
    train_proba = _align_proba_to_classes(
        model.classes_, model.predict_proba(bundle.X_train), bundle.class_names
    )
    train_metrics = compute_classification_metrics(
        bundle.y_train, train_pred, bundle.class_names, train_proba
    )

    cv_summary = run_cross_validation(
        model_factory=model_factory,
        X=bundle.X_train,
        y=bundle.y_train,
        classes=bundle.class_names,
        n_splits=n_splits,
        deduplicate=True,
        seed=seed,
    )

    test_metrics: dict[str, Any] | None = None
    eval_y_true = bundle.y_train
    eval_y_pred = train_pred
    if bundle.X_test is not None and bundle.y_test is not None:
        test_pred = model.predict(bundle.X_test)
        test_proba = _align_proba_to_classes(
            model.classes_, model.predict_proba(bundle.X_test), bundle.class_names
        )
        test_metrics = compute_classification_metrics(
            bundle.y_test, test_pred, bundle.class_names, test_proba
        )
        eval_y_true = bundle.y_test
        eval_y_pred = test_pred
        eval_base_X = bundle.X_test
    else:
        eval_base_X = bundle.X_train

    dropout_20_accs: list[float] = []
    dropout_40_accs: list[float] = []
    for run_i in range(5):
        X_d20 = perturb_symptoms(eval_base_X, drop_rate=0.20, noise_rate=0.01, seed=seed + run_i)
        X_d40 = perturb_symptoms(eval_base_X, drop_rate=0.40, noise_rate=0.01, seed=seed + 100 + run_i)
        dropout_20_accs.append(float(np.mean(model.predict(X_d20) == eval_y_true)))
        dropout_40_accs.append(float(np.mean(model.predict(X_d40) == eval_y_true)))

    d20_mean = float(np.mean(dropout_20_accs))
    d40_mean = float(np.mean(dropout_40_accs))
    audit = assess_overfitting_and_quality(
        train_acc=float(train_metrics["accuracy"]),
        cv_summary=cv_summary,
        test_metrics=test_metrics,
        dropout_20_acc=d20_mean,
    )

    record = {
        "model_name": name,
        "train_time_ms": round(train_time_ms, 2),
        "train_metrics": {
            k: v for k, v in train_metrics.items() if k not in ("confusion_matrix", "per_class")
        },
        "cv_summary": {
            k: v for k, v in cv_summary.items() if k != "oof_metrics"
        },
        "test_metrics": (
            {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
            if test_metrics is not None
            else None
        ),
        "dropout_20_accuracy": round(d20_mean, 6),
        "dropout_40_accuracy": round(d40_mean, 6),
        "overfitting_audit": audit,
    }
    return record, model, eval_y_true, eval_y_pred


def _save_alpha_plot(sweep: list[dict[str, Any]], out_path: Path, model_label: str) -> None:
    alphas = [row["alpha"] for row in sweep]
    train_err = [1.0 - row["mean_train_accuracy"] for row in sweep]
    cv_err = [row["cv_error_rate"] for row in sweep]
    cv_loss = [row["cv_log_loss"] or 0.0 for row in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(alphas, train_err, marker="o", label="Train Error Rate")
    ax1.plot(alphas, cv_err, marker="s", label="Deduplicated CV Error Rate")
    ax1.set_xscale("log")
    ax1.set_xlabel("Laplace Smoothing (alpha)")
    ax1.set_ylabel("Error Rate")
    ax1.set_title(f"{model_label}: Train vs CV Error Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(alphas, cv_loss, marker="^", color="tab:red", label="CV Log-Loss")
    ax2.set_xscale("log")
    ax2.set_xlabel("Laplace Smoothing (alpha)")
    ax2.set_ylabel("Cross-Entropy Log-Loss")
    ax2.set_title(f"{model_label}: Calibration (CV Log-Loss)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_confusion_matrix_plot(
    cm: np.ndarray, classes: list[str], out_path: Path, title: str
) -> None:
    size = max(7.5, min(16.0, len(classes) * 0.35))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90, fontsize=7 if len(classes) > 15 else 9)
    ax.set_yticklabels(classes, fontsize=7 if len(classes) > 15 else 9)
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("True Disease")
    ax.set_title(title)

    if len(classes) <= 20:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7,
                    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_comparison_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    names = [r["model_name"] for r in records]
    cv_accs = [r["cv_summary"]["mean_val_accuracy"] * 100.0 for r in records]
    d20_accs = [r["dropout_20_accuracy"] * 100.0 for r in records]
    d40_accs = [r["dropout_40_accuracy"] * 100.0 for r in records]
    test_accs = [
        (r["test_metrics"]["accuracy"] * 100.0)
        if r["test_metrics"] is not None
        else r["train_metrics"]["accuracy"] * 100.0
        for r in records
    ]

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 1.5 * width, test_accs, width, label="Test (or Train) %")
    ax.bar(x - 0.5 * width, cv_accs, width, label="Deduplicated CV %")
    ax.bar(x + 0.5 * width, d20_accs, width, label="20% Dropout + Noise %")
    ax.bar(x + 1.5 * width, d40_accs, width, label="40% Dropout + Noise %")

    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("VitalAI Model Comparison & Symptom-Dropout Robustness")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_full_evaluation(
    train_path: Path,
    test_path: Path | None,
    artifact_dir: Path,
    n_splits: int = 5,
    fit_prior: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune smoothing, compare NB variants vs KNN, audit overfitting, and save plots."""
    bundle = load_dataset(train_path, test_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mnb_tuning = tune_alpha(
        bundle.X_train,
        bundle.y_train,
        bundle.feature_names,
        bundle.class_names,
        model_cls=MultinomialNaiveBayes,
        n_splits=n_splits,
        fit_prior=fit_prior,
        seed=seed,
    )
    bnb_tuning = tune_alpha(
        bundle.X_train,
        bundle.y_train,
        bundle.feature_names,
        bundle.class_names,
        model_cls=BernoulliNaiveBayes,
        n_splits=n_splits,
        fit_prior=fit_prior,
        seed=seed,
    )

    mnb_alpha = float(mnb_tuning["best_alpha"])
    bnb_alpha = float(bnb_tuning["best_alpha"])

    mnb_rec, _, eval_true, eval_pred = _evaluate_single_model(
        f"MultinomialNB(alpha={mnb_alpha})",
        lambda: MultinomialNaiveBayes(alpha=mnb_alpha, fit_prior=fit_prior),
        bundle,
        n_splits,
        seed,
    )
    bnb_rec, _, bnb_true, bnb_pred = _evaluate_single_model(
        f"BernoulliNB(alpha={bnb_alpha})",
        lambda: BernoulliNaiveBayes(alpha=bnb_alpha, fit_prior=fit_prior),
        bundle,
        n_splits,
        seed,
    )
    knn_rec, _, _, _ = _evaluate_single_model(
        "CosineKNN(k=5)",
        lambda: CosineKNNClassifier(k=5, deduplicate=True),
        bundle,
        n_splits,
        seed,
    )

    candidates = [
        ("multinomial", mnb_alpha, mnb_rec, eval_true, eval_pred),
        ("bernoulli", bnb_alpha, bnb_rec, bnb_true, bnb_pred),
    ]
    winner_type, winner_alpha, winner_rec, winner_true, winner_pred = min(
        candidates,
        key=lambda c: (
            c[2]["overfitting_audit"]["cv_error_rate"],
            c[2]["overfitting_audit"]["test_error_rate"]
            if c[2]["overfitting_audit"]["test_error_rate"] is not None
            else 0.0,
            -c[2]["dropout_20_accuracy"],
        ),
    )

    cm = np.array(
        compute_classification_metrics(winner_true, winner_pred, bundle.class_names)[
            "confusion_matrix"
        ],
        dtype=np.int64,
    )

    _save_alpha_plot(
        mnb_tuning["sweep"],
        artifact_dir / "alpha_tuning_curve.png",
        "MultinomialNB",
    )
    _save_confusion_matrix_plot(
        cm,
        bundle.class_names,
        artifact_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — {winner_rec['model_name']}",
    )
    _save_comparison_plot(
        [mnb_rec, bnb_rec, knn_rec],
        artifact_dir / "model_comparison.png",
    )

    report = {
        "dataset_diagnostics": bundle.diagnostics,
        "recommended_model": {
            "model_type": winner_type,
            "alpha": winner_alpha,
            "fit_prior": fit_prior,
            "overfitting_audit": winner_rec["overfitting_audit"],
        },
        "alpha_tuning": {
            "multinomial": mnb_tuning,
            "bernoulli": bnb_tuning,
        },
        "model_comparison": [mnb_rec, bnb_rec, knn_rec],
        "plots": {
            "alpha_tuning_curve": str(artifact_dir / "alpha_tuning_curve.png"),
            "confusion_matrix": str(artifact_dir / "confusion_matrix.png"),
            "model_comparison": str(artifact_dir / "model_comparison.png"),
        },
    }
    report_path = artifact_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VitalAI ML models for accuracy and overfitting."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-file", type=str, default="Training.csv"
    )
    parser.add_argument(
        "--test-file", type=str, default="Testing.csv"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("backend/artifacts")
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--uniform-prior", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.data_dir / args.train_file
    test_path = args.data_dir / args.test_file
    if not train_path.exists():
        raise SystemExit(
            f"[ERROR] '{train_path}' not found.\n"
            "Place the Kaggle Training.csv and Testing.csv inside data/ first, "
            "or pass --data-dir / --train-file pointing to your CSV."
        )

    report = run_full_evaluation(
        train_path=train_path,
        test_path=test_path if test_path.exists() else None,
        artifact_dir=args.artifact_dir,
        n_splits=args.n_splits,
        fit_prior=not args.uniform_prior,
        seed=args.seed,
    )
    rec = report["recommended_model"]
    audit = rec["overfitting_audit"]
    print(f"Recommended Model : {rec['model_type']} (alpha={rec['alpha']})")
    print(f"Audit Status      : {audit['status']}")
    print(f"Train Accuracy    : {audit['train_accuracy']:.4f}")
    print(f"CV Error Rate     : {audit['cv_error_rate']:.4f} (gap={audit['cv_generalization_gap']:.4f})")
    if audit["test_error_rate"] is not None:
        print(f"Test Error Rate   : {audit['test_error_rate']:.4f} (gap={audit['test_generalization_gap']:.4f})")
    print(f"20% Dropout Error : {audit['dropout_20_error_rate']:.4f}")
    print(f"Saved report      : {args.artifact_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
