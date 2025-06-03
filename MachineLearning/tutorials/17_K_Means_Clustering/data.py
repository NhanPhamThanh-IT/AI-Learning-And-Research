from sklearn.datasets import make_blobs
import numpy as np

def load_data():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.2, random_state=42)
    return X 