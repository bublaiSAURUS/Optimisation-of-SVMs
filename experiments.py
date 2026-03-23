import numpy as np
from optimisers import smo, smo_modified
from examples import demo, fetchdata
from utils import plotter

class SVM:
    def __init__(self, kernel = 'linear', C = 1.0, max_iter = 200, tol = 1e-3, sv_tol = 1e-8):
        if kernel == 'linear':
            self._kernel = self.linear_kernel
        elif kernel == 'gaussian':
            self._kernel = self.gaussian_kernel 
        self._C = C
        self._max_iter = max_iter
        self._tol = tol
        self._sv_tol = sv_tol

    def linear_kernel(self, x1, x2, b=0):
        return x1 @ x2.T + b

    def gaussian_kernel(self, x1, x2, sigma=1):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-(np.linalg.norm(x1-x2,2))**2/(2*sigma**2))
        elif(np.ndim(x1)>1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2)>1):
            return np.exp(-(np.linalg.norm(x1-x2, 2, axis=1)**2)/(2*sigma**2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1 :
            return np.exp(-(np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], 2, axis = 2) ** 2)/(2*sigma**2))
        return 0.

    def compute_kernel_support_vectors(self, X):
        return self._kernel(X, self.support_vectors)

    def fit(self, X, y):
        self.solver = smo.SMO(X, y, C = self._C, kernel = self._kernel, max_iter=self._max_iter, tol=self._tol)
        self.alphas, self.b = self.solver.fit()
        self.alpha_history = self.solver.alpha_history
        sv = self.alphas > self._sv_tol
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.alpha_i = self.alphas[sv] * self.support_labels

        # plotter.plot_obj_func(self.solver.obj_history)
        # plotter.plot_Q_conv(self.alpha_history)
        # plotter.plot_errors(self.solver.obj_history)
        # plotter.plot_boundary(X, y, self.solver)

    def fit_modified(self, X, y):
        self.solver = smo_modified.SMO_MOD(X, y, C=1, kernel = self._kernel, tol = self._tol)
        self.alphas = self.solver.fit()
        # plotter.plot_obj_func(self.solver.obj_history)
        # plotter.plot_errors(self.solver.obj_history)
        # plotter.plot_Q_conv(self.solver.alpha_history, p = 1)
        plotter.plot_moons_boundary(X, y, self.solver)



    def predict(self, X):
        kernel_support_vectors = self.compute_kernel_support_vectors(X)
        decision = kernel_support_vectors @ self.alpha_i + self.b
        return decision

    
if  __name__ == "__main__":
    svm = SVM(kernel = "gaussian")
    # X, y = demo.create_demo()
    # X, y = fetchdata.get_data()
    X, y = fetchdata.make_toydata(n_samples = 600)
    svm.fit_modified(X, y)
    # print(svm.alpha_history)
    # plotter.plot_demo(svm, X, y, filename = "demo.png")


