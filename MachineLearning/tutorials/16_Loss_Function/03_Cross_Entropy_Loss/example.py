import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Binary Cross Entropy Example
y_true_bin = np.array([1, 0, 1, 1])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.7])
bce = binary_cross_entropy(y_true_bin, y_pred_bin)
print(f"Binary Cross Entropy: {bce}")

# Categorical Cross Entropy Example
y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_cat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
cce = categorical_cross_entropy(y_true_cat, y_pred_cat)
print(f"Categorical Cross Entropy: {cce}") 