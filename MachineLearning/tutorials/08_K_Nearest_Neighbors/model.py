import numpy as np
from sklearn.metrics import pairwise_distances

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        dists = pairwise_distances(X, self.X_train)
        neighbors = np.argsort(dists, axis=1)[:, :self.k]
        y_pred = []
        for idx in neighbors:
            labels, counts = np.unique(self.y_train[idx], return_counts=True)
            y_pred.append(labels[np.argmax(counts)])
        return np.array(y_pred)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) 