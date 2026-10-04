# SMO-SVM

A compact, installable research library for training binary support vector
machines with **Sequential Minimal Optimization (SMO)**. It keeps two distinct
working-set strategies:

- `algorithm="platt"` implements John Platt's original 1998 SMO heuristics.
- `algorithm="keerthi"` implements the two-threshold modification proposed by
  Keerthi, Shevade, Bhattacharyya, and Murthy.

The project favors readable algorithmic code, observable convergence state, and
reproducible experiments over production-scale performance.

## Why two implementations?

Both solvers optimize the same soft-margin SVM dual,

```text
maximize    sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j)
subject to  0 <= alpha_i <= C
            sum_i alpha_i y_i = 0,
```

but choose pairs of dual variables differently:

| Solver | Working-set strategy | Threshold handling |
| --- | --- | --- |
| Platt | Examine-all/non-bound passes, then maximize the error difference | One explicit prediction intercept |
| Keerthi | Most-violating pairs derived from index sets `I0`–`I4` | Maintains `b_up` and `b_low` and recovers the intercept at termination |

The shared analytical two-variable update lives in a small common base class;
the search and stopping logic remains separate so the algorithms can be studied
and compared directly.

## Installation

SMO-SVM requires Python 3.10 or newer and NumPy.

```bash
git clone <repository-url>
cd Optimisation-of-SVMs
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and plotting:

```bash
python -m pip install -e ".[dev,plot]"
```

## Quick start

```python
import numpy as np

from smo_svm import SMOClassifier

x = np.array([
    [-2.0, -1.0],
    [-1.0, -2.0],
    [1.0, 2.0],
    [2.0, 1.0],
])
y = np.array(["negative", "negative", "positive", "positive"])

model = SMOClassifier(
    algorithm="platt",
    C=1.0,
    kernel="linear",
    random_state=0,
).fit(x, y)

print(model.predict(x))
print(model.decision_function([[0.5, 1.0]]))
print(model.score(x, y))
```

Switch to the modified algorithm without changing the estimator API:

```python
model = SMOClassifier(
    algorithm="keerthi",
    C=1.0,
    kernel="rbf",
    gamma="scale",
    tol=1e-3,
    max_iter=1_000,
).fit(x, y)
```

Arbitrary binary labels are encoded internally as `{-1, +1}` and converted
back by `predict`. Multiclass classification is intentionally out of scope.

## Estimator parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `algorithm` | `"keerthi"` | `"platt"` or `"keerthi"` |
| `C` | `1.0` | Soft-margin box constraint |
| `kernel` | `"rbf"` | `"linear"`, `"rbf"`, `"gaussian"`, or a callable |
| `gamma` | `"scale"` | RBF coefficient; positive float, `"scale"`, or `"auto"` |
| `tol` | `1e-3` | KKT stopping tolerance |
| `eps` | `1e-8` | Numerical threshold for alpha updates and bounds |
| `max_iter` | `1000` | Maximum number of outer passes |
| `support_tolerance` | `1e-8` | Minimum alpha counted as a support vector |
| `random_state` | `None` | Seed for Platt's randomized fallback scans |
| `track_history` | `False` | Store every alpha vector after a successful step |

Both built-in kernels precompute a dense Gram matrix during fitting. A custom
kernel must accept two 2-D arrays and return a matrix with shape
`(len(left), len(right))`:

```python
def polynomial(left, right):
    return (1.0 + left @ right.T) ** 2

model = SMOClassifier(kernel=polynomial).fit(x, y)
```

## Learned attributes

After `fit`, the most useful attributes are:

- `alphas_`, `intercept_`, and `dual_coef_`;
- `support_`, `support_vectors_`, and `n_support_`;
- `n_iter_` and `converged_`;
- `objective_history_` and, when enabled, `alpha_history_`;
- `equality_residual_`, which reports `|sum(alpha_i y_i)|`;
- `coef_` for the built-in linear kernel.

If the iteration budget is exhausted, the estimator retains the current model,
sets `converged_ = False`, and emits `ConvergenceWarning`.

## Diagnostics and plotting

The package exposes reusable research diagnostics:

```python
from smo_svm import dual_objective, equality_residual, gram_condition_number
```

Optional plotting helpers return Matplotlib axes rather than calling `show`, so
callers control presentation and file output:

```python
import matplotlib.pyplot as plt

from smo_svm.plotting import plot_decision_boundary, plot_objective_history

plot_objective_history(model)
plot_decision_boundary(model, x, y)
plt.show()
```

## Comparing the algorithms

The bundled example fits both solvers to the same deterministic dataset:

```bash
python examples/compare_algorithms.py
```

It reports training accuracy, support-vector count, outer iterations, and
convergence status for each strategy.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
python -m build
```

The numerical tests cover both solvers on linearly separable data and an RBF
XOR problem. They also check dual feasibility, objective monotonicity, label
round-tripping, kernels, validation, and convergence warnings.

## Package layout

```text
src/smo_svm/
├── __init__.py
├── classifier.py       Public estimator API and validation
├── diagnostics.py      Dual-objective and conditioning utilities
├── kernels.py          Linear/RBF kernels and gamma resolution
├── plotting.py         Optional research visualizations
└── solvers/
    ├── _base.py        Shared two-variable optimization mechanics
    ├── platt.py        Original 1998 working-set strategy
    └── keerthi.py      Two-threshold modified strategy
examples/
└── compare_algorithms.py
tests/
├── test_classifier.py
└── test_kernels.py
```

## Scope and limitations

- Binary C-SVM classification only; there is no multiclass wrapper or SVR.
- Dense in-memory Gram matrices require `O(n^2)` storage. This makes the code
  easy to inspect but unsuitable for very large datasets.
- The implementation is designed for research, teaching, and numerical
  comparison—not as a replacement for optimized libraries such as LIBSVM.
- Inputs should be scaled when feature magnitudes differ substantially.
- No software license has been declared for this repository.

## References

1. J. C. Platt, *Sequential Minimal Optimization: A Fast Algorithm for Training
   Support Vector Machines*, Microsoft Research Technical Report MSR-TR-98-14,
   1998. [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)
2. S. S. Keerthi, S. K. Shevade, C. Bhattacharyya, and K. R. K. Murthy,
   *Improvements to Platt's SMO Algorithm for SVM Classifier Design*, Neural
   Computation 13(3), 637–649, 2001.
   [DOI](https://doi.org/10.1162/089976601300014493)
