import numpy as np
from optimisers import smo, smo_modified
from examples import demo, fetchdata
from utils import plotter
import time
import psutil
import os

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

    def linear_kernel(self, x1, x2):
        return x1 @ x2.T

    def gaussian_kernel(self, x1, x2, gamma=0.5):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-gamma*(np.linalg.norm(x1-x2,2))**2)
        elif(np.ndim(x1)>1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2)>1):
            return np.exp(-gamma*(np.linalg.norm(x1-x2, 2, axis=1)**2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1 :
            return np.exp(-gamma*(np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], 2, axis = 2) ** 2))
        return 0.
    
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def fit(self, X, y):
        self.solver = smo_modified.SMO_MOD(X, y, C=1, kernel = self._kernel, tol = self._tol)
        self.alphas = self.solver.fit()
        # plotter.plot_obj_func(self.solver.obj_history)
        # plotter.plot_errors(self.solver.obj_history)
        # plotter.plot_Q_conv(self.solver.alpha_history, p = 1)
        # plotter.plot_moons_boundary(X, y, self.solver)


def check_condition(X, y, K):
    Q = np.outer(y, y) * K
    eigenval = np.linalg.eigvalsh(Q)
    min_eigenvalue = np.min(eigenval)
    print(f"Minimum eigenvalue = {min_eigenvalue}")
    cond_number = np.linalg.cond(Q)

    print(f"Condition Number: {cond_number:.2e}")

    if cond_number > 1e12:
        print("Warning: Matrix is ill-conditioned.")
    else:
        print("Matrix is well-conditioned.")

def check_degeneracy(X, y, K, alphas, b, C, tol = 1e-6):
    Q = np.outer(y, y) * K 
    Qa = Q @ alphas
    grad = Qa - 1
    violations = 0
    total_checked = 0
    
    for i in range(len(alphas)):
        if alphas[i] < tol or abs(alphas[i] - C) < tol:
            total_checked += 1
            
            val = grad[i] + b * y[i]
            
            if abs(val) < tol:
                violations += 1
    
    print(f"Checked {total_checked} boundary points")
    print(f"Strict complementarity violations: {violations}")            
    
if  __name__ == "__main__":

    # X, y = fetchdata.get_data() # To get Spambase
    X, y = fetchdata.make_toydata(n_samples = 600, make_plot=True)
    svm = SVM(kernel = "gaussian")
    K = svm.gaussian_kernel(X, X)
    check_condition(X, y, K)
    svm.fit(X, y)
    b = (svm.solver.b_up + svm.solver.b_low)/2.0
    check_degeneracy(X, y, K, svm.solver.alphas, b, svm._C)

    # n_sample_list = [x for x in range(100, 2100, 100)]
    # time_elapsed = []
    # space_list = []
    
    # for n_samples in n_sample_list:
    #     X, y = fetchdata.make_toydata(n_samples = n_samples)
    #     svm = SVM(kernel = "gaussian")
    #     start_time = time.perf_counter()
    #     svm.fit(X, y)
    #     end_time = time.perf_counter()
    #     peak_mem = svm.get_memory_usage()
    #     elapsed = end_time - start_time
    #     time_elapsed.append(elapsed)
    #     space_list.append(peak_mem)
    #     print(f"Size: {n_samples} | Time: {elapsed:.4f}s | Memory: {peak_mem:.4f}Mb")
    
    # plotter.plot_timecomplexity(n_sample_list, time_elapsed)
    # plotter.plot_spacecomplexity(n_sample_list, space_list)

    # X, y = demo.create_demo()  # To run the demo
    # plotter.plot_demo(svm, X, y, filename = "demo.png") # Plot for demo


