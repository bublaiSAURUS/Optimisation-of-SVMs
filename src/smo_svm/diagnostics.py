"""Diagnostics for the SVM dual optimization problem."""

from __future__ import annotations

import numpy as np

from .kernels import FloatArray


def dual_objective(
    alphas: FloatArray,
    targets: FloatArray,
    gram: FloatArray,
) -> float:
    """Return the maximized soft-margin SVM dual objective."""
    weighted = alphas * targets
    return float(np.sum(alphas) - 0.5 * weighted @ gram @ weighted)


def equality_residual(alphas: FloatArray, targets: FloatArray) -> float:
    """Return ``|sum(alpha_i y_i)|`` for the dual equality constraint."""
    return float(abs(np.dot(alphas, targets)))


def gram_condition_number(gram: FloatArray) -> float:
    """Return the 2-norm condition number of a Gram matrix."""
    return float(np.linalg.cond(gram))


__all__ = ["dual_objective", "equality_residual", "gram_condition_number"]
