# Placeholder for running the Gradient Descent example 

import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear_data
from model import predict, mean_squared_error, mse_gradient

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Implements the Gradient Descent algorithm for linear regression.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        learning_rate (float): The step size for parameter updates.
        n_iterations (int): The number of iterations to run Gradient Descent.

    Returns:
        tuple: A tuple containing:
            - w (np.ndarray): The learned weights.
            - b (float): The learned bias.
            - cost_history (list): List of MSE values at each iteration.
    """
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    b = 0
    cost_history = []

    for i in range(n_iterations):
        y_pred = predict(X, w, b)
        cost = mean_squared_error(y, y_pred)
        cost_history.append(cost)

        dw, db = mse_gradient(X, y, y_pred)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Cost: {cost:.4f}")

    return w, b, cost_history

if __name__ == '__main__':
    # 1. Generate Data
    X, y = generate_linear_data(n_samples=100, noise=30)

    # 2. Run Gradient Descent
    learning_rate = 0.0001
    n_iterations = 10000
    w, b, cost_history = gradient_descent(X, y, learning_rate, n_iterations)

    print("\nFinal Parameters:")
    print("Weights (w):", w)
    print("Bias (b):", b)

    # 3. Make Predictions with learned parameters
    y_pred_final = predict(X, w, b)

    # 4. Visualize Results
    plt.figure(figsize=(12, 5))

    # Plot Cost History
    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Cost Function Progress (Gradient Descent)")

    # Plot Data and Regression Line
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, marker='.', label='Actual Data')
    plt.plot(X, y_pred_final, color='red', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression with Gradient Descent")
    plt.legend()

    plt.tight_layout()
    plt.show() 