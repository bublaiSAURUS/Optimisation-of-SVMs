"""A small, research-oriented binary SVM classifier trained with SMO."""

from __future__ import annotations

import warnings

import numpy as np

from .diagnostics import equality_residual
from .kernels import FloatArray, Kernel, resolve_kernel
from .solvers import KeerthiSMOSolver, PlattSMOSolver


class ConvergenceWarning(UserWarning):
    """Emitted when an SMO solver exhausts its outer-iteration budget."""


class SMOClassifier:
    """Binary C-SVM classifier with two selectable SMO algorithms.

    Parameters mirror familiar SVM estimators where practical. ``algorithm``
    chooses either Platt's original 1998 heuristic or the two-threshold
    modification by Keerthi et al. (2001).
    """

    def __init__(
        self,
        *,
        algorithm: str = "keerthi",
        C: float = 1.0,
        kernel: str | Kernel = "rbf",
        gamma: float | str = "scale",
        tol: float = 1e-3,
        eps: float = 1e-8,
        max_iter: int = 1_000,
        support_tolerance: float = 1e-8,
        random_state: int | None = None,
        track_history: bool = False,
    ) -> None:
        self.algorithm = algorithm
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.support_tolerance = support_tolerance
        self.random_state = random_state
        self.track_history = track_history

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return constructor parameters for estimator compatibility."""
        del deep
        return {
            "algorithm": self.algorithm,
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "tol": self.tol,
            "eps": self.eps,
            "max_iter": self.max_iter,
            "support_tolerance": self.support_tolerance,
            "random_state": self.random_state,
            "track_history": self.track_history,
        }

    def set_params(self, **params: object) -> SMOClassifier:
        """Set constructor parameters and return this estimator."""
        valid = self.get_params()
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def _validate_hyperparameters(self) -> None:
        if self.algorithm not in {"platt", "keerthi"}:
            raise ValueError("algorithm must be 'platt' or 'keerthi'")
        if not np.isfinite(self.C) or self.C <= 0.0:
            raise ValueError("C must be a positive finite value")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be a positive finite value")
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be a positive finite value")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 0.0 <= self.support_tolerance < self.C:
            raise ValueError("support_tolerance must be in [0, C)")

    @staticmethod
    def _validate_training_data(
        x: FloatArray,
        y: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must have shape (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape (n_samples,)")
        if x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("At least two samples and one feature are required")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("SMOClassifier supports exactly two classes")
        encoded = np.where(y == classes[1], 1.0, -1.0)
        return x, encoded, classes

    def fit(self, x: FloatArray, y: FloatArray) -> SMOClassifier:
        """Fit a binary SVM and return this estimator."""
        self._validate_hyperparameters()
        x, encoded_y, self.classes_ = self._validate_training_data(x, y)
        self.n_features_in_ = x.shape[1]
        self.kernel_, self.gamma_ = resolve_kernel(self.kernel, self.gamma, x)
        gram = np.asarray(self.kernel_(x, x), dtype=float)
        expected_shape = (x.shape[0], x.shape[0])
        if gram.shape != expected_shape:
            raise ValueError(
                f"kernel(x, x) returned {gram.shape}; expected {expected_shape}"
            )
        if not np.isfinite(gram).all():
            raise ValueError("The kernel matrix must contain only finite values")
        if not np.allclose(gram, gram.T, atol=1e-10, rtol=1e-8):
            raise ValueError("The training kernel matrix must be symmetric")
        solver_type = (
            PlattSMOSolver if self.algorithm == "platt" else KeerthiSMOSolver
        )
        solver = solver_type(
            c=float(self.C),
            tol=float(self.tol),
            eps=float(self.eps),
            max_iter=int(self.max_iter),
            random_state=self.random_state,
            track_history=self.track_history,
        )
        result = solver.solve(gram, encoded_y)
        self.alphas_ = result.alphas
        self.intercept_ = np.array([result.intercept], dtype=float)
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.objective_history_ = np.asarray(result.objective_history)
        self.alpha_history_ = result.alpha_history
        self.equality_residual_ = equality_residual(self.alphas_, encoded_y)
        self.support_ = np.flatnonzero(self.alphas_ > self.support_tolerance)
        self.support_vectors_ = x[self.support_].copy()
        self.support_labels_ = encoded_y[self.support_].copy()
        self.dual_coef_ = (
            self.alphas_[self.support_] * self.support_labels_
        )[None, :]
        self.n_support_ = np.array(
            [
                np.count_nonzero(self.support_labels_ < 0.0),
                np.count_nonzero(self.support_labels_ > 0.0),
            ],
            dtype=int,
        )
        self.x_fit_ = x.copy()
        self.y_fit_ = encoded_y
        if self.kernel == "linear":
            self.coef_ = (self.alphas_ * encoded_y) @ x
        if not self.converged_:
            warnings.warn(
                f"{self.algorithm} SMO reached max_iter={self.max_iter}",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("Call fit() before using this estimator")

    def decision_function(self, x: FloatArray) -> FloatArray:
        """Return signed distances up to the SVM's kernel-space scaling."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x must have shape (n_samples, {self.n_features_in_})"
            )
        gram = np.asarray(self.kernel_(self.support_vectors_, x), dtype=float)
        expected_shape = (self.support_.shape[0], x.shape[0])
        if gram.shape != expected_shape:
            raise ValueError(
                f"kernel(support_vectors, x) returned {gram.shape}; "
                f"expected {expected_shape}"
            )
        return (self.dual_coef_ @ gram).reshape(-1) + self.intercept_[0]

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict labels in the same representation supplied to ``fit``."""
        indices = (self.decision_function(x) >= 0.0).astype(int)
        return self.classes_[indices]

    def score(self, x: FloatArray, y: FloatArray) -> float:
        """Return mean classification accuracy."""
        y = np.asarray(y)
        predictions = self.predict(x)
        if y.shape != predictions.shape:
            raise ValueError("y has an incompatible shape")
        return float(np.mean(predictions == y))

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "support_vectors_")


__all__ = ["ConvergenceWarning", "SMOClassifier"]
