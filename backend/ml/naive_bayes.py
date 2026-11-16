"""Self-written Naive Bayes classifiers in NumPy with explainability & serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PredictionAnalysis:
    """Detailed inference output passed from ML engine to Expert System."""

    top_predictions: list[dict[str, Any]]
    normalized_entropy: float
    confidence_margin: float
    symptom_contributions: list[dict[str, Any]]
    discriminating_followups: list[dict[str, Any]]
    alternative_diseases: list[dict[str, Any]] = None  # type: ignore[assignment]
    symptom_causes: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.alternative_diseases is None:
            self.alternative_diseases = []
        if self.symptom_causes is None:
            self.symptom_causes = []


def _log_sum_exp(scores: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp along axis 1."""
    max_scores = np.max(scores, axis=1, keepdims=True)
    stabilized = scores - max_scores
    return max_scores + np.log(np.sum(np.exp(stabilized), axis=1, keepdims=True))


class _BaseNaiveBayes:
    """Shared training, inference, explainability, and persistence logic."""

    model_type: str = "base"

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("Laplace smoothing parameter 'alpha' must be > 0.")
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.classes_: np.ndarray = np.array([], dtype=object)
        self.feature_names_: list[str] = []
        self.class_counts_: np.ndarray = np.array([], dtype=np.float64)
        self.class_log_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.array([], dtype=np.float64)

    def _validate_fitted(self) -> None:
        if self.classes_.size == 0 or self.feature_log_prob_.size == 0:
            raise RuntimeError("Model must be fitted before running inference.")

    def _prepare_fit_inputs(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=object)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X_arr.shape[0] == 0:
            raise ValueError("Cannot fit Naive Bayes on an empty dataset.")
        if np.any(X_arr < 0):
            raise ValueError("Feature matrix X must be non-negative.")

        if feature_names is not None:
            if len(feature_names) != X_arr.shape[1]:
                raise ValueError("feature_names length must match X.shape[1].")
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X_arr.shape[1])]

        classes = sorted(set(str(label) for label in y_arr))
        self.classes_ = np.array(classes, dtype=object)
        n_classes = len(classes)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        y_idx = np.array([class_to_idx[str(label)] for label in y_arr], dtype=np.int64)
        counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        self.class_counts_ = counts

        if self.fit_prior:
            self.class_log_prior_ = np.log(counts / float(X_arr.shape[0]))
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes), dtype=np.float64)

        return X_arr, y_idx

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _symptom_score_matrix(self) -> np.ndarray:
        """Return per-class feature log-odds/weights used for explainability."""
        return self.feature_log_prob_

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected X with shape (n_samples, {len(self.feature_names_)}), got {X_arr.shape}."
            )
        jll = self._joint_log_likelihood(X_arr)
        return jll - _log_sum_exp(jll)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        probs = np.exp(log_probs)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.where(row_sums > 0, row_sums, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        jll = self._joint_log_likelihood(np.asarray(X, dtype=np.float64))
        best_indices = np.argmax(jll, axis=1)
        return self.classes_[best_indices]

    def _relative_symptom_strength(self) -> np.ndarray:
        """Compute per-disease normalized symptom prominence in [0, 1]."""
        probs = np.exp(self.feature_log_prob_)
        row_max = np.max(probs, axis=1, keepdims=True)
        return probs / np.where(row_max > 0, row_max, 1.0)

    def _explain_disease(
        self,
        cls_idx: int,
        x_arr: np.ndarray,
        rel_strength: np.ndarray,
    ) -> tuple[list[str], list[str]]:
        """Return matched supporting symptoms and absent hallmark symptoms for a class."""
        active_indices = np.where(x_arr > 0)[0]
        absent_indices = np.where(x_arr == 0)[0]

        matched = [
            (i, float(rel_strength[cls_idx, i]))
            for i in active_indices
            if rel_strength[cls_idx, i] >= 0.25
        ]
        matched.sort(key=lambda pair: pair[1], reverse=True)

        missing = [
            (i, float(rel_strength[cls_idx, i]))
            for i in absent_indices
            if rel_strength[cls_idx, i] >= 0.40
        ]
        missing.sort(key=lambda pair: pair[1], reverse=True)

        return (
            [self.feature_names_[i] for i, _ in matched],
            [self.feature_names_[i] for i, _ in missing[:4]],
        )

    def predict_detailed(self, x: np.ndarray, top_k: int = 3) -> PredictionAnalysis:
        """Run single-patient inference with uncertainty & symptom explainability."""
        self._validate_fitted()
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if x_arr.shape[0] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, got {x_arr.shape[0]}."
            )
        if np.sum(x_arr > 0) == 0:
            raise ValueError("At least one active symptom is required for prediction.")

        probs = self.predict_proba(x_arr[None, :])[0]
        ranked_idx = np.argsort(probs)[::-1]
        k = max(1, min(int(top_k), len(self.classes_)))
        rel_strength = self._relative_symptom_strength()

        top_predictions = []
        for rank, idx in enumerate(ranked_idx[:k]):
            supporting, missing = self._explain_disease(int(idx), x_arr, rel_strength)
            top_predictions.append(
                {
                    "disease": str(self.classes_[idx]),
                    "probability": float(probs[idx]),
                    "rank": rank + 1,
                    "supporting_symptoms": supporting,
                    "missing_hallmarks": missing,
                }
            )

        positive = probs[probs > 1e-15]
        raw_entropy = float(-np.sum(positive * np.log(positive)))
        max_entropy = float(np.log(len(self.classes_))) if len(self.classes_) > 1 else 1.0
        normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        margin = (
            float(probs[ranked_idx[0]] - probs[ranked_idx[1]])
            if len(ranked_idx) > 1
            else 1.0
        )

        weights = self._symptom_score_matrix()
        top1_idx = int(ranked_idx[0])
        active_indices = np.where(x_arr > 0)[0]

        if len(self.classes_) > 1:
            other_mask = np.ones(len(self.classes_), dtype=bool)
            other_mask[top1_idx] = False
            baseline = np.mean(weights[other_mask], axis=0)
        else:
            baseline = np.zeros_like(weights[top1_idx])

        contributions: list[dict[str, Any]] = []
        for feat_idx in active_indices:
            lift = float(weights[top1_idx, feat_idx] - baseline[feat_idx])
            contributions.append(
                {
                    "symptom": self.feature_names_[feat_idx],
                    "log_likelihood_lift": round(lift, 4),
                }
            )
        contributions.sort(key=lambda item: item["log_likelihood_lift"], reverse=True)

        discriminating: list[dict[str, Any]] = []
        if len(ranked_idx) > 1:
            top2_idx = int(ranked_idx[1])
            absent_indices = np.where(x_arr == 0)[0]
            for feat_idx in absent_indices:
                delta = float(weights[top1_idx, feat_idx] - weights[top2_idx, feat_idx])
                discriminating.append(
                    {
                        "symptom": self.feature_names_[feat_idx],
                        "separation_score": round(abs(delta), 4),
                        "favors": str(
                            self.classes_[top1_idx]
                            if delta >= 0
                            else self.classes_[top2_idx]
                        ),
                    }
                )
            discriminating.sort(key=lambda item: item["separation_score"], reverse=True)

        alternative_diseases: list[dict[str, Any]] = []
        for cls_idx in range(len(self.classes_)):
            if cls_idx == top1_idx:
                continue
            supporting, missing = self._explain_disease(cls_idx, x_arr, rel_strength)
            if supporting:
                alternative_diseases.append(
                    {
                        "disease": str(self.classes_[cls_idx]),
                        "probability": float(probs[cls_idx]),
                        "matched_symptoms": supporting,
                        "missing_hallmarks": missing,
                    }
                )
        alternative_diseases.sort(
            key=lambda item: (len(item["matched_symptoms"]), item["probability"]),
            reverse=True,
        )

        symptom_causes: list[dict[str, Any]] = []
        for feat_idx in active_indices:
            matching_cls = [
                c_i
                for c_i in range(len(self.classes_))
                if rel_strength[c_i, feat_idx] >= 0.35
            ]
            matching_cls.sort(
                key=lambda c_i: (float(rel_strength[c_i, feat_idx]), float(probs[c_i])),
                reverse=True,
            )
            symptom_causes.append(
                {
                    "symptom": self.feature_names_[feat_idx],
                    "possible_diseases": [str(self.classes_[c_i]) for c_i in matching_cls[:6]],
                }
            )

        return PredictionAnalysis(
            top_predictions=top_predictions,
            normalized_entropy=float(normalized_entropy),
            confidence_margin=margin,
            symptom_contributions=contributions[:6],
            discriminating_followups=discriminating[:5],
            alternative_diseases=alternative_diseases[:15],
            symptom_causes=symptom_causes,
        )

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def save(self, artifact_dir: Path, prefix: str = "nb_model") -> Path:
        """Persist model weights (.npz) and metadata (.json) safely without pickle."""
        self._validate_fitted()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        npz_path = artifact_dir / f"{prefix}.npz"
        meta_path = artifact_dir / f"{prefix}_meta.json"

        arrays: dict[str, np.ndarray] = {
            "class_counts": self.class_counts_,
            "class_log_prior": self.class_log_prior_,
            "feature_log_prob": self.feature_log_prob_,
        }
        arrays.update(self._extra_arrays())
        np.savez(npz_path, **arrays)

        metadata = {
            "model_type": self.model_type,
            "alpha": self.alpha,
            "fit_prior": self.fit_prior,
            "classes": [str(c) for c in self.classes_.tolist()],
            "feature_names": self.feature_names_,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return npz_path

    @classmethod
    def load(cls, artifact_dir: Path, prefix: str = "nb_model") -> _BaseNaiveBayes:
        """Load a saved Naive Bayes classifier from .npz and metadata JSON."""
        npz_path = artifact_dir / f"{prefix}.npz"
        meta_path = artifact_dir / f"{prefix}_meta.json"
        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing saved model artifacts in {artifact_dir} with prefix '{prefix}'."
            )

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        model_type = metadata["model_type"]
        registry: dict[str, type[_BaseNaiveBayes]] = {
            "multinomial": MultinomialNaiveBayes,
            "bernoulli": BernoulliNaiveBayes,
        }
        if model_type not in registry:
            raise ValueError(f"Unsupported model_type '{model_type}' in metadata.")

        target_cls = registry[model_type]
        model = target_cls(alpha=metadata["alpha"], fit_prior=metadata["fit_prior"])
        model.classes_ = np.array(metadata["classes"], dtype=object)
        model.feature_names_ = list(metadata["feature_names"])

        with np.load(npz_path, allow_pickle=False) as data:
            model.class_counts_ = data["class_counts"]
            model.class_log_prior_ = data["class_log_prior"]
            model.feature_log_prob_ = data["feature_log_prob"]
            if model_type == "bernoulli":
                assert isinstance(model, BernoulliNaiveBayes)
                model.feature_neg_log_prob_ = data["feature_neg_log_prob"]

        return model


class MultinomialNaiveBayes(_BaseNaiveBayes):
    """Self-written Multinomial Naive Bayes classifier for symptom count/presence vectors."""

    model_type = "multinomial"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MultinomialNaiveBayes:
        X_arr, y_idx = self._prepare_fit_inputs(X, y, feature_names)
        n_classes = len(self.classes_)
        n_features = X_arr.shape[1]

        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        for cls_i in range(n_classes):
            feature_counts[cls_i] = X_arr[y_idx == cls_i].sum(axis=0)

        smoothed_fc = feature_counts + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return X @ self.feature_log_prob_.T + self.class_log_prior_


class BernoulliNaiveBayes(_BaseNaiveBayes):
    """Self-written Multivariate Bernoulli Naive Bayes modeling present and absent symptoms."""

    model_type = "bernoulli"

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        super().__init__(alpha=alpha, fit_prior=fit_prior)
        self.feature_neg_log_prob_: np.ndarray = np.array([], dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> BernoulliNaiveBayes:
        X_arr, y_idx = self._prepare_fit_inputs(X, y, feature_names)
        X_bin = (X_arr > 0).astype(np.float64)
        n_classes = len(self.classes_)
        n_features = X_bin.shape[1]

        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        for cls_i in range(n_classes):
            feature_counts[cls_i] = X_bin[y_idx == cls_i].sum(axis=0)

        probs = (feature_counts + self.alpha) / (
            self.class_counts_[:, None] + 2.0 * self.alpha
        )
        self.feature_log_prob_ = np.log(probs)
        self.feature_neg_log_prob_ = np.log(1.0 - probs)
        return self

    def _symptom_score_matrix(self) -> np.ndarray:
        return self.feature_log_prob_ - self.feature_neg_log_prob_

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X_bin = (X > 0).astype(np.float64)
        diff = self.feature_log_prob_ - self.feature_neg_log_prob_
        neg_sum = self.feature_neg_log_prob_.sum(axis=1)
        return X_bin @ diff.T + neg_sum + self.class_log_prior_

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        return {"feature_neg_log_prob": self.feature_neg_log_prob_}
