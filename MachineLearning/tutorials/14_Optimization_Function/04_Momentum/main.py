import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear_data
from model import predict, mean_squared_error, mse_gradient

def gradient_descent_with_momentum(X, y, learning_rate=0.01, momentum=0.9, n_iterations=1000):
    """
    Implements the Gradient Descent with Momentum algorithm for linear regression.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        learning_rate (float): The step size for parameter updates.
        momentum (float): The momentum term (beta). Controls how much of the previous velocity is retained.
        n_iterations (int): The number of iterations to run.

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

    # Initialize velocity
    vw = np.zeros((n_features, 1))
    vb = 0

    for i in range(n_iterations):
        y_pred = predict(X, w, b)
        cost = mean_squared_error(y, y_pred)
        cost_history.append(cost)

        dw, db = mse_gradient(X, y, y_pred)

        # Update velocity
        vw = momentum * vw + learning_rate * dw
        vb = momentum * vb + learning_rate * db

        # Update parameters using velocity
        w -= vw
        b -= vb

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Cost: {cost:.4f}")

    return w, b, cost_history

if __name__ == '__main__':
    # 1. Generate Data
    X, y = generate_linear_data(n_samples=100, noise=30)

    # 2. Run Gradient Descent with Momentum
    learning_rate = 0.0001
    momentum = 0.9
    n_iterations = 10000
    w, b, cost_history = gradient_descent_with_momentum(X, y, learning_rate, momentum, n_iterations)

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
    plt.title("Cost Function Progress (Momentum)")

    # Plot Data and Regression Line
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, marker='.', label='Actual Data')
    # Sort X for plotting the line correctly
    sort_indices = np.argsort(X[:, 0])
    X_sorted = X[sort_indices]
    y_pred_final_sorted = predict(X_sorted, w, b)
    plt.plot(X_sorted, y_pred_final_sorted, color='red', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression with Momentum")
    plt.legend()

    plt.tight_layout()
    plt.show() 