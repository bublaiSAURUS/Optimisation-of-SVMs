"""Keerthi et al.'s two-threshold modification of SMO."""

from __future__ import annotations

import numpy as np

from ..kernels import FloatArray
from ._base import BaseSMOSolver, SolverResult


class KeerthiSMOSolver(BaseSMOSolver):
    """Solve the binary SVM dual using ``b_up`` and ``b_low`` thresholds."""

    def _index_masks(self) -> tuple[np.ndarray, ...]:
        non_bound = (self.alphas > self.eps) & (
            self.alphas < self.c - self.eps
        )
        positive = self.targets > 0.0
        at_lower = self.alphas <= self.eps
        at_upper = self.alphas >= self.c - self.eps
        i1 = positive & at_lower
        i2 = ~positive & at_upper
        i3 = positive & at_upper
        i4 = ~positive & at_lower
        return non_bound, i1, i2, i3, i4

    def _refresh_thresholds(self) -> None:
        i0, i1, i2, i3, i4 = self._index_masks()
        upper_candidates = i0 | i1 | i2
        lower_candidates = i0 | i3 | i4
        upper_indices = np.flatnonzero(upper_candidates)
        lower_indices = np.flatnonzero(lower_candidates)
        self.i_up = int(upper_indices[np.argmin(self.function_values[upper_indices])])
        self.i_low = int(lower_indices[np.argmax(self.function_values[lower_indices])])
        self.b_up = float(self.function_values[self.i_up])
        self.b_low = float(self.function_values[self.i_low])

    def _take_step(self, i1: int, i2: int) -> bool:
        if i1 == i2:
            return False
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = self.targets[i1], self.targets[i2]
        value1 = self.function_values[i1]
        value2 = self.function_values[i2]
        lower, upper = self._bounds(i1, i2)
        if upper - lower <= self.eps:
            return False
        k11 = self.gram[i1, i1]
        k12 = self.gram[i1, i2]
        k22 = self.gram[i2, i2]
        eta = 2.0 * k12 - k11 - k22
        if eta < 0.0:
            candidate2 = alpha2 - y2 * (value1 - value2) / eta
            candidate2 = float(np.clip(candidate2, lower, upper))
        else:
            candidate2 = self._select_bound(i1, i2, lower, upper)
        if abs(candidate2 - alpha2) < self.eps * (
            candidate2 + alpha2 + self.eps
        ):
            return False
        sign = y1 * y2
        candidate1 = alpha1 + sign * (alpha2 - candidate2)
        delta1 = candidate1 - alpha1
        delta2 = candidate2 - alpha2
        self.alphas[i1] = candidate1
        self.alphas[i2] = candidate2
        self.function_values += (
            y1 * delta1 * self.gram[:, i1]
            + y2 * delta2 * self.gram[:, i2]
        )
        self._refresh_thresholds()
        self._record_step()
        return True

    def _examine(self, i2: int) -> bool:
        value2 = self.function_values[i2]
        i0, i1, i2_mask, i3, i4 = self._index_masks()
        is_upper_candidate = bool((i0 | i1 | i2_mask)[i2])
        is_lower_candidate = bool((i0 | i3 | i4)[i2])
        first_violation = self.b_low - value2
        second_violation = value2 - self.b_up
        selected: int | None = None
        if is_upper_candidate and first_violation > 2.0 * self.tol:
            selected = self.i_low
        if is_lower_candidate and second_violation > 2.0 * self.tol:
            if selected is None or second_violation > first_violation:
                selected = self.i_up
        if selected is None:
            return False
        return self._take_step(selected, i2)

    def solve(self, gram: FloatArray, targets: FloatArray) -> SolverResult:
        """Optimize the dual problem for a precomputed Gram matrix."""
        self._initialize(gram, targets)
        # F_i = sum_j(alpha_j y_j K_ij) - y_i. The intercept is recovered
        # from the final upper and lower thresholds.
        self.function_values = -targets.copy()
        self._refresh_thresholds()
        examine_all = True
        num_changed = 0
        n_iter = 0
        while (num_changed > 0 or examine_all) and n_iter < self.max_iter:
            num_changed = 0
            if examine_all:
                for index in range(self.n_samples):
                    num_changed += self._examine(index)
            else:
                while self.b_low - self.b_up > 2.0 * self.tol:
                    if not self._take_step(self.i_up, self.i_low):
                        break
                    num_changed += 1
            n_iter += 1
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        intercept = -0.5 * (self.b_up + self.b_low)
        converged = self.b_low - self.b_up <= 2.0 * self.tol
        return SolverResult(
            alphas=self.alphas.copy(),
            intercept=float(intercept),
            n_iter=n_iter,
            converged=converged,
            objective_history=tuple(self.objective_history),
            alpha_history=tuple(self.alpha_history),
        )


__all__ = ["KeerthiSMOSolver"]
