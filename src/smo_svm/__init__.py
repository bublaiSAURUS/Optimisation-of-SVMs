"""Research implementations of sequential minimal optimization for SVMs."""

from .classifier import ConvergenceWarning, SMOClassifier
from .diagnostics import dual_objective, equality_residual, gram_condition_number
from .kernels import linear_kernel, rbf_kernel
from .solvers import KeerthiSMOSolver, PlattSMOSolver

__version__ = "0.1.0"

__all__ = [
    "ConvergenceWarning",
    "KeerthiSMOSolver",
    "PlattSMOSolver",
    "SMOClassifier",
    "dual_objective",
    "equality_residual",
    "gram_condition_number",
    "linear_kernel",
    "rbf_kernel",
]
