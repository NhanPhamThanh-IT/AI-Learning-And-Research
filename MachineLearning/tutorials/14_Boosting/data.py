from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test 