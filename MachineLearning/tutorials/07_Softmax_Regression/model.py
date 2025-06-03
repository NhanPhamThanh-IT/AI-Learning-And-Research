import numpy as np

class SoftmaxRegression:
    def __init__(self, num_features, num_classes):
        self.W = np.zeros((num_features, num_classes))
        self.b = np.zeros(num_classes)
        self.num_classes = num_classes

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, epochs=1000, lr=0.1):
        y_onehot = np.eye(self.num_classes)[y]
        for epoch in range(epochs):
            logits = X @ self.W + self.b
            probs = self.softmax(logits)
            grad_W = X.T @ (probs - y_onehot) / X.shape[0]
            grad_b = np.mean(probs - y_onehot, axis=0)
            self.W -= lr * grad_W
            self.b -= lr * grad_b

    def predict(self, X):
        logits = X @ self.W + self.b
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) 