from data import load_data
from model import DecisionTree

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = DecisionTree(max_depth=3)
    model.fit(X_train, y_train)
    acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}") 