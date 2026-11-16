"""CLI entrypoint to train, anti-overfit validate, and serialize the VitalAI ML model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.ml.data_loader import load_dataset
from backend.ml.evaluate_model import _evaluate_single_model, run_full_evaluation
from backend.ml.naive_bayes import BernoulliNaiveBayes, MultinomialNaiveBayes


def train_and_save(
    train_path: Path,
    test_path: Path | None,
    artifact_dir: Path,
    model_type: str = "auto",
    alpha: float | None = None,
    n_splits: int = 5,
    fit_prior: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run evaluation-backed hyperparameter selection, fit final model, and save artifacts."""
    bundle = load_dataset(train_path, test_path)
    eval_report = run_full_evaluation(
        train_path=train_path,
        test_path=test_path,
        artifact_dir=artifact_dir,
        n_splits=n_splits,
        fit_prior=fit_prior,
        seed=seed,
    )

    if model_type == "auto":
        selected_type = str(eval_report["recommended_model"]["model_type"])
    else:
        selected_type = model_type.lower()

    if selected_type not in ("multinomial", "bernoulli"):
        raise ValueError("model_type must be 'auto', 'multinomial', or 'bernoulli'.")

    if alpha is None:
        selected_alpha = float(eval_report["alpha_tuning"][selected_type]["best_alpha"])
    else:
        selected_alpha = float(alpha)

    cls = MultinomialNaiveBayes if selected_type == "multinomial" else BernoulliNaiveBayes
    eval_rec, model, _, _ = _evaluate_single_model(
        f"{cls.__name__}(alpha={selected_alpha})",
        lambda: cls(alpha=selected_alpha, fit_prior=fit_prior),
        bundle,
        n_splits=n_splits,
        seed=seed,
    )
    npz_path = model.save(artifact_dir, prefix="nb_model")

    summary = {
        "model_path": str(npz_path),
        "selected_model_type": selected_type,
        "selected_alpha": selected_alpha,
        "fit_prior": fit_prior,
        "n_features": len(bundle.feature_names),
        "n_classes": len(bundle.class_names),
        "feature_display_names": bundle.feature_display_names,
        "overfitting_audit": eval_rec["overfitting_audit"],
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate VitalAI Naive Bayes model against overfitting."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-file", type=str, default="Training.csv")
    parser.add_argument("--test-file", type=str, default="Testing.csv")
    parser.add_argument("--artifact-dir", type=Path, default=Path("backend/artifacts"))
    parser.add_argument(
        "--model-type",
        choices=("auto", "multinomial", "bernoulli"),
        default="auto",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--uniform-prior", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.data_dir / args.train_file
    test_path = args.data_dir / args.test_file
    if not train_path.exists():
        raise SystemExit(
            f"[ERROR] '{train_path}' not found.\n"
            "Download Kaggle's 'Disease Prediction Using Machine Learning' Training.csv & Testing.csv "
            "into data/ before running training."
        )

    summary = train_and_save(
        train_path=train_path,
        test_path=test_path if test_path.exists() else None,
        artifact_dir=args.artifact_dir,
        model_type=args.model_type,
        alpha=args.alpha,
        n_splits=args.n_splits,
        fit_prior=not args.uniform_prior,
        seed=args.seed,
    )
    audit = summary["overfitting_audit"]
    print(f"Trained & Saved   : {summary['model_path']}")
    print(f"Selected Model    : {summary['selected_model_type']} (alpha={summary['selected_alpha']})")
    print(f"Overfit/Err Gate  : {audit['status']}")
    print(f"Audit Notes       : {'; '.join(audit['notes'])}")


if __name__ == "__main__":
    main()
