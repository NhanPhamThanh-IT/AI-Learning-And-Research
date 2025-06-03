import numpy as np
from data import generate_data
from model import SimpleLogisticRegression

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

if __name__ == "__main__":
    # Generate data
    X_train, X_test, y_train, y_test = generate_data()

    # Initialize and train the model
    model = SimpleLogisticRegression(learning_rate=0.001, n_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    acc = accuracy(y_test, predictions)
    print(f"Accuracy: {acc:.4f}") 