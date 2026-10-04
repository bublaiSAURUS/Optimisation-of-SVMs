"""Compare Platt and Keerthi SMO on the same synthetic binary problem."""

from __future__ import annotations

import numpy as np

from smo_svm import SMOClassifier


def make_dataset(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    negative = rng.normal(loc=(-1.0, -1.0), scale=0.65, size=(60, 2))
    positive = rng.normal(loc=(1.0, 1.0), scale=0.65, size=(60, 2))
    x = np.vstack((negative, positive))
    y = np.concatenate((-np.ones(negative.shape[0]), np.ones(positive.shape[0])))
    return x, y


def main() -> None:
    x, y = make_dataset()
    for algorithm in ("platt", "keerthi"):
        model = SMOClassifier(
            algorithm=algorithm,
            C=1.0,
            kernel="rbf",
            gamma="scale",
            random_state=0,
            track_history=True,
        ).fit(x, y)
        print(
            f"{algorithm:7s} | accuracy={model.score(x, y):.3f} "
            f"| support={model.support_.size:3d} | iterations={model.n_iter_:3d} "
            f"| converged={model.converged_}"
        )


if __name__ == "__main__":
    main()
