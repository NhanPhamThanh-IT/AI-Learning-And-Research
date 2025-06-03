# Placeholder for running the Stochastic Gradient Descent example 

import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear_data
from model import predict, mse_gradient_sgd

def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=100):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for linear regression.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        learning_rate (float): The step size for parameter updates.
        n_epochs (int): The number of epochs to run SGD.

    Returns:
        tuple: A tuple containing:
            - w (np.ndarray): The learned weights.
            - b (float): The learned bias.
            - cost_history (list): List of cost values (averaged per epoch or taken periodically).
    """
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    b = 0
    cost_history = []

    for epoch in range(n_epochs):
        # Shuffle data at the beginning of each epoch
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        epoch_cost = 0
        for i in range(n_samples):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]

            y_pred = predict(xi, w, b)

            # Calculate gradient for the current sample
            dw, db = mse_gradient_sgd(xi, yi, y_pred)

            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db

            # Optional: Calculate cost for this sample (not usually done in practice for plotting SGD)
            # epoch_cost += mean_squared_error_sgd(yi[0,0], y_pred[0,0]) # Note: need to adjust dimensions

        # Calculate average cost for the epoch for plotting
        y_pred_epoch = predict(X, w, b)
        cost_epoch = np.mean((y - y_pred_epoch)**2) # Using full batch MSE for plotting clarity
        cost_history.append(cost_epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Cost: {cost_epoch:.4f}")

    return w, b, cost_history

if __name__ == '__main__':
    # 1. Generate Data
    X, y = generate_linear_data(n_samples=100, noise=30)

    # 2. Run Stochastic Gradient Descent
    learning_rate = 0.001
    n_epochs = 100
    w, b, cost_history = stochastic_gradient_descent(X, y, learning_rate, n_epochs)

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
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Cost Function Progress (SGD)")

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
    plt.title("Linear Regression with SGD")
    plt.legend()

    plt.tight_layout()
    plt.show() 