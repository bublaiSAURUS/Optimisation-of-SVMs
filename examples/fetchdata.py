from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np



def get_data():
    dataset = fetch_ucirepo(id=94)
    X = dataset.data.features.values
    y_raw = dataset.data.targets.values.flatten()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.where(y_raw == 1, 1.0, -1.0).astype(float)
    return X_scaled, y

def make_toydata(n_samples, make_plot = False):
    X, y = make_moons(n_samples= n_samples, noise=0.3, random_state=42)
    y = np.where(y == 0, -1, 1).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if make_plot:
        plt.title("Moons dataset")
        plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='red', label='Class 1')
        plt.scatter(X_scaled[y == -1, 0], X_scaled[y == -1, 1], color='blue', label='Class -1')
        plt.legend()
        plt.savefig("dataset.png")
        plt.close()
    return X_scaled, y