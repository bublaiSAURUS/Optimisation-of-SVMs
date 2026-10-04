"""Kernel functions and kernel resolution utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Kernel: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]


def linear_kernel(x: FloatArray, y: FloatArray) -> FloatArray:
    """Return the linear Gram matrix ``x @ y.T``."""
    return np.asarray(x, dtype=float) @ np.asarray(y, dtype=float).T


def rbf_kernel(
    x: FloatArray,
    y: FloatArray,
    *,
    gamma: float = 1.0,
) -> FloatArray:
    """Return the radial-basis-function Gram matrix."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    squared_distances = (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return np.exp(-gamma * squared_distances)


def resolve_gamma(gamma: float | str, x: FloatArray) -> float:
    """Resolve scikit-learn-style ``gamma`` settings for the RBF kernel."""
    if isinstance(gamma, str):
        if gamma == "auto":
            return 1.0 / x.shape[1]
        if gamma == "scale":
            variance = float(np.var(x))
            return 1.0 / (x.shape[1] * variance) if variance > 0.0 else 1.0
        raise ValueError("gamma must be a positive float, 'scale', or 'auto'")
    resolved = float(gamma)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("gamma must be a positive finite value")
    return resolved


def resolve_kernel(
    kernel: str | Kernel,
    gamma: float | str,
    x: FloatArray,
) -> tuple[Kernel, float | None]:
    """Return a validated kernel callable and its resolved gamma."""
    if callable(kernel):
        return kernel, None
    if kernel == "linear":
        return linear_kernel, None
    if kernel in {"rbf", "gaussian"}:
        resolved_gamma = resolve_gamma(gamma, x)

        def configured_rbf(left: FloatArray, right: FloatArray) -> FloatArray:
            return rbf_kernel(left, right, gamma=resolved_gamma)

        return configured_rbf, resolved_gamma
    raise ValueError("kernel must be 'linear', 'rbf', 'gaussian', or a callable")


__all__ = ["FloatArray", "Kernel", "linear_kernel", "rbf_kernel"]
