"""Shared mechanics for binary SMO solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..diagnostics import dual_objective
from ..kernels import FloatArray


@dataclass(frozen=True)
class SolverResult:
    """State returned by a converged or budget-limited solver."""

    alphas: FloatArray
    intercept: float
    n_iter: int
    converged: bool
    objective_history: tuple[float, ...]
    alpha_history: tuple[FloatArray, ...]


class BaseSMOSolver:
    """Base class containing mechanics shared by both algorithms."""

    def __init__(
        self,
        *,
        c: float,
        tol: float,
        eps: float,
        max_iter: int,
        random_state: int | None,
        track_history: bool,
    ) -> None:
        self.c = c
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)
        self.track_history = track_history

    def _initialize(self, gram: FloatArray, targets: FloatArray) -> None:
        self.gram = gram
        self.targets = targets
        self.n_samples = targets.shape[0]
        self.alphas = np.zeros(self.n_samples, dtype=float)
        self.objective_history: list[float] = [0.0]
        self.alpha_history: list[FloatArray] = [self.alphas.copy()]
        self.n_steps = 0

    def _bounds(self, i1: int, i2: int) -> tuple[float, float]:
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        if self.targets[i1] != self.targets[i2]:
            return max(0.0, alpha2 - alpha1), min(
                self.c, self.c + alpha2 - alpha1
            )
        return max(0.0, alpha1 + alpha2 - self.c), min(
            self.c, alpha1 + alpha2
        )

    def _select_bound(
        self,
        i1: int,
        i2: int,
        lower: float,
        upper: float,
    ) -> float:
        """Choose the better feasible bound when curvature is non-positive."""
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        sign = self.targets[i1] * self.targets[i2]

        def objective_at(candidate2: float) -> float:
            candidate = self.alphas.copy()
            candidate[i2] = candidate2
            candidate[i1] = alpha1 + sign * (alpha2 - candidate2)
            return dual_objective(candidate, self.targets, self.gram)

        lower_objective = objective_at(lower)
        upper_objective = objective_at(upper)
        if lower_objective > upper_objective + self.eps:
            return lower
        if upper_objective > lower_objective + self.eps:
            return upper
        return alpha2

    def _record_step(self) -> None:
        self.n_steps += 1
        self.objective_history.append(
            dual_objective(self.alphas, self.targets, self.gram)
        )
        if self.track_history:
            self.alpha_history.append(self.alphas.copy())


__all__ = ["BaseSMOSolver", "SolverResult"]
