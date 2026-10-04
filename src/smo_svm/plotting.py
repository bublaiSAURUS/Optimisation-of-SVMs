"""Optional plotting helpers for fitted SMO classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .classifier import SMOClassifier
from .kernels import FloatArray


def plot_objective_history(
    model: SMOClassifier,
    *,
    ax: Any = None,
) -> Any:
    """Plot the maximized dual objective after each successful SMO step."""
    import matplotlib.pyplot as plt

    if not hasattr(model, "objective_history_"):
        raise RuntimeError("The classifier must be fitted before plotting")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(model.objective_history_.size), model.objective_history_)
    ax.set_xlabel("Successful SMO step")
    ax.set_ylabel("Dual objective")
    ax.set_title(f"{model.algorithm.capitalize()} SMO convergence")
    return ax


def plot_decision_boundary(
    model: SMOClassifier,
    x: FloatArray,
    y: FloatArray,
    *,
    ax: Any = None,
    resolution: int = 200,
) -> Any:
    """Plot a fitted binary decision boundary for two-dimensional inputs."""
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("plot_decision_boundary requires x with two features")
    if resolution < 10:
        raise ValueError("resolution must be at least 10")
    if ax is None:
        _, ax = plt.subplots()
    padding = 0.1 * np.maximum(np.ptp(x, axis=0), 1.0)
    lower = x.min(axis=0) - padding
    upper = x.max(axis=0) + padding
    xx, yy = np.meshgrid(
        np.linspace(lower[0], upper[0], resolution),
        np.linspace(lower[1], upper[1], resolution),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    scores = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, scores, levels=[-np.inf, 0.0, np.inf], alpha=0.15)
    ax.contour(xx, yy, scores, levels=[0.0], colors="black")
    for label in model.classes_:
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], label=str(label), edgecolors="black")
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="black",
        label="support vectors",
    )
    ax.set_title(f"{model.algorithm.capitalize()} SMO decision boundary")
    ax.legend()
    return ax


__all__ = ["plot_decision_boundary", "plot_objective_history"]
