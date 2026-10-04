"""John Platt's original 1998 SMO working-set strategy."""

from __future__ import annotations

import numpy as np

from ..kernels import FloatArray
from ._base import BaseSMOSolver, SolverResult


class PlattSMOSolver(BaseSMOSolver):
    """Solve the binary SVM dual with Platt's original heuristics."""

    def _take_step(self, i1: int, i2: int) -> bool:
        if i1 == i2:
            return False
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = self.targets[i1], self.targets[i2]
        error1, error2 = self.errors[i1], self.errors[i2]
        lower, upper = self._bounds(i1, i2)
        if upper - lower <= self.eps:
            return False
        k11 = self.gram[i1, i1]
        k12 = self.gram[i1, i2]
        k22 = self.gram[i2, i2]
        eta = k11 + k22 - 2.0 * k12
        if eta > 0.0:
            candidate2 = alpha2 + y2 * (error1 - error2) / eta
            candidate2 = float(np.clip(candidate2, lower, upper))
        else:
            candidate2 = self._select_bound(i1, i2, lower, upper)
        if abs(candidate2 - alpha2) < self.eps * (
            candidate2 + alpha2 + self.eps
        ):
            return False
        sign = y1 * y2
        candidate1 = alpha1 + sign * (alpha2 - candidate2)
        old_intercept = self.intercept
        intercept1 = (
            old_intercept
            - error1
            - y1 * (candidate1 - alpha1) * k11
            - y2 * (candidate2 - alpha2) * k12
        )
        intercept2 = (
            old_intercept
            - error2
            - y1 * (candidate1 - alpha1) * k12
            - y2 * (candidate2 - alpha2) * k22
        )
        if self.eps < candidate1 < self.c - self.eps:
            self.intercept = intercept1
        elif self.eps < candidate2 < self.c - self.eps:
            self.intercept = intercept2
        else:
            self.intercept = 0.5 * (intercept1 + intercept2)
        delta1 = candidate1 - alpha1
        delta2 = candidate2 - alpha2
        self.alphas[i1] = candidate1
        self.alphas[i2] = candidate2
        self.errors += (
            y1 * delta1 * self.gram[:, i1]
            + y2 * delta2 * self.gram[:, i2]
            + self.intercept
            - old_intercept
        )
        self._record_step()
        return True

    def _examine(self, i2: int) -> bool:
        error2 = self.errors[i2]
        alpha2 = self.alphas[i2]
        violation = error2 * self.targets[i2]
        if not (
            (violation < -self.tol and alpha2 < self.c - self.eps)
            or (violation > self.tol and alpha2 > self.eps)
        ):
            return False
        non_bound = np.flatnonzero(
            (self.alphas > self.eps) & (self.alphas < self.c - self.eps)
        )
        if non_bound.size > 1:
            differences = np.abs(self.errors[non_bound] - error2)
            i1 = int(non_bound[np.argmax(differences)])
            if self._take_step(i1, i2):
                return True
        for i1 in self.rng.permutation(non_bound):
            if self._take_step(int(i1), i2):
                return True
        for i1 in self.rng.permutation(self.n_samples):
            if self._take_step(int(i1), i2):
                return True
        return False

    def solve(self, gram: FloatArray, targets: FloatArray) -> SolverResult:
        """Optimize the dual problem for a precomputed Gram matrix."""
        self._initialize(gram, targets)
        self.intercept = 0.0
        self.errors = -targets.copy()
        examine_all = True
        num_changed = 0
        n_iter = 0
        while (num_changed > 0 or examine_all) and n_iter < self.max_iter:
            num_changed = 0
            indices = (
                np.arange(self.n_samples)
                if examine_all
                else np.flatnonzero(
                    (self.alphas > self.eps)
                    & (self.alphas < self.c - self.eps)
                )
            )
            for index in indices:
                num_changed += self._examine(int(index))
            n_iter += 1
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        converged = not (num_changed > 0 or examine_all)
        return SolverResult(
            alphas=self.alphas.copy(),
            intercept=float(self.intercept),
            n_iter=n_iter,
            converged=converged,
            objective_history=tuple(self.objective_history),
            alpha_history=tuple(self.alpha_history),
        )


__all__ = ["PlattSMOSolver"]
