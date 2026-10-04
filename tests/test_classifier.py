import unittest
import warnings

import numpy as np

from smo_svm import ConvergenceWarning, SMOClassifier


class SMOClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
        )
        self.y = np.array(["negative", "negative", "positive", "positive"])

    def test_both_algorithms_fit_a_linear_problem(self) -> None:
        for algorithm in ("platt", "keerthi"):
            with self.subTest(algorithm=algorithm):
                model = SMOClassifier(
                    algorithm=algorithm,
                    kernel="linear",
                    C=10.0,
                    max_iter=1_000,
                    random_state=0,
                    track_history=True,
                ).fit(self.x, self.y)
                np.testing.assert_array_equal(model.predict(self.x), self.y)
                self.assertLess(model.equality_residual_, 1e-8)
                self.assertGreater(model.support_.size, 0)
                self.assertEqual(model.coef_.shape, (2,))
                self.assertTrue(
                    np.all(np.diff(model.objective_history_) >= -1e-8)
                )

    def test_rbf_kernel_separates_xor(self) -> None:
        x = np.array(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        )
        y = np.array([-1, 1, 1, -1])
        for algorithm in ("platt", "keerthi"):
            with self.subTest(algorithm=algorithm):
                model = SMOClassifier(
                    algorithm=algorithm,
                    kernel="rbf",
                    gamma=1.0,
                    C=10.0,
                    max_iter=1_000,
                    random_state=0,
                ).fit(x, y)
                np.testing.assert_array_equal(model.predict(x), y)

    def test_iteration_budget_emits_warning(self) -> None:
        model = SMOClassifier(
            algorithm="platt",
            kernel="linear",
            max_iter=1,
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(self.x, self.y)
        self.assertTrue(
            any(issubclass(item.category, ConvergenceWarning) for item in caught)
        )

    def test_rejects_multiclass_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            SMOClassifier().fit(self.x[:3], np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
