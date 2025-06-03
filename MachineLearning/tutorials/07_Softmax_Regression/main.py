import numpy as np
from data import load_data
from model import SoftmaxRegression

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = SoftmaxRegression(num_features=X_train.shape[1], num_classes=len(np.unique(y_train)))
    model.fit(X_train, y_train, epochs=1000, lr=0.1)
    acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}") 