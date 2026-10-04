import unittest

import numpy as np

from smo_svm import linear_kernel, rbf_kernel


class KernelTests(unittest.TestCase):
    def test_linear_kernel(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(linear_kernel(x, x), x @ x.T)

    def test_rbf_kernel_is_symmetric_with_unit_diagonal(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 1.0]])
        gram = rbf_kernel(x, x, gamma=0.5)
        np.testing.assert_allclose(gram, gram.T)
        np.testing.assert_allclose(np.diag(gram), np.ones(x.shape[0]))


if __name__ == "__main__":
    unittest.main()
