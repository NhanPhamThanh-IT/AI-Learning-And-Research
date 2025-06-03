import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc}") 