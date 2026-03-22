from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
import numpy as np

def get_data():
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, -1, 1).astype(float)
    return X, y

def make_toydata(n_samples):
    X, y = make_blobs(n_samples= n_samples, centers=2, cluster_std= 3.0, n_features=2, random_state=42)
    y = np.where(y == 0, -1, 1).astype(float)
    return X, y