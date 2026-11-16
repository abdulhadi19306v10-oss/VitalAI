"""Self-written Cosine-Similarity KNN baseline classifier in NumPy."""

from __future__ import annotations

import numpy as np

from backend.ml.data_loader import deduplicate_patterns


class CosineKNNClassifier:
    """Distance-weighted K-Nearest Neighbors baseline for ML model comparison."""

    def __init__(self, k: int = 5, deduplicate: bool = True) -> None:
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = int(k)
        self.deduplicate = bool(deduplicate)
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._X_train_norm: np.ndarray = np.array([], dtype=np.float64)
        self._y_train_idx: np.ndarray = np.array([], dtype=np.int64)

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return np.divide(X, norms, out=np.zeros_like(X, dtype=np.float64), where=norms > 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CosineKNNClassifier:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=object)
        if self.deduplicate:
            X_clean, y_clean, _ = deduplicate_patterns(X_arr.astype(np.int64), y_arr)
            X_arr = X_clean.astype(np.float64)
            y_arr = y_clean

        classes = sorted(set(str(label) for label in y_arr))
        self.classes_ = np.array(classes, dtype=object)
        cls_to_idx = {cls: i for i, cls in enumerate(classes)}

        self._X_train_norm = self._l2_normalize(X_arr)
        self._y_train_idx = np.array(
            [cls_to_idx[str(label)] for label in y_arr], dtype=np.int64
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._X_train_norm.size == 0:
            raise RuntimeError("CosineKNNClassifier must be fitted before inference.")
        X_norm = self._l2_normalize(np.asarray(X, dtype=np.float64))
        sims = X_norm @ self._X_train_norm.T

        n_samples = X_norm.shape[0]
        n_classes = len(self.classes_)
        k_eff = min(self.k, self._X_train_norm.shape[0])
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_samples):
            row_sims = sims[i]
            top_idx = np.argpartition(row_sims, -k_eff)[-k_eff:]
            weights = np.maximum(row_sims[top_idx], 0.0) + 1e-9
            neighbor_classes = self._y_train_idx[top_idx]
            class_scores = np.bincount(
                neighbor_classes, weights=weights, minlength=n_classes
            ).astype(np.float64)
            total = class_scores.sum()
            probs[i] = class_scores / total if total > 0 else (1.0 / n_classes)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
