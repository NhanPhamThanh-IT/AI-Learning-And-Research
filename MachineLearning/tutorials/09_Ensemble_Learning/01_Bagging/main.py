from data import load_data
from model import BaggingModel

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = BaggingModel()
    model.fit(X_train, y_train)
    acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}") 