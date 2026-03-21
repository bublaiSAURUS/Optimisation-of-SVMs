import numpy as np
from optimisers import smo
from examples import demo, fetchdata
from utils import plotter

class SVM:
    def __init__(self, kernel = 'linear', C = 1.0, max_iter = 200, tol = 1e-5, sv_tol = 1e-8):
        if kernel == 'linear':
            self._kernel = self.linear_kernel
        elif kernel  =='polynomial':
            self._kernel = self.polynomial_kernel
        elif kernel == 'gaussian':
            self._kernel = self.gaussian_kernel 
        self._C = C
        self._max_iter = max_iter
        self._tol = tol
        self._sv_tol = sv_tol

    def linear_kernel(self, x1, x2, b=0):
        return x1 @ x2.T + b
    
    def polynomial_kernel(self, x1, x2, b = 0, p = 3):
        return (x1 @ x2.T + b)**p

    def gaussian_kernel(self, x1, x2, sigma = 1):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * (x1 @ x2.T)
        
        return np.exp(-sq_dist / (2 * (sigma**2)))

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
        plotter.plot_boundary(X, y, self.solver)


    def predict(self, X):
        kernel_support_vectors = self.compute_kernel_support_vectors(X)
        decision = kernel_support_vectors @ self.alpha_i + self.b
        return decision

    
if  __name__ == "__main__":
    svm = SVM(kernel = "linear")
    # X, y = demo.create_demo()
    # X, y = fetchdata.get_data()
    X, y = fetchdata.make_toydata(n_samples = 100)
    svm.fit(X, y)
    # print(svm.alpha_history)
    # plotter.plot_demo(svm, X, y, filename = "demo.png")


