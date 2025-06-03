import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: The MSE.
    """
    return np.mean((y_true - y_pred)**2)

def mse_gradient(X, y_true, y_pred):
    """
    Calculates the gradient of the Mean Squared Error with respect to the weights and bias 
    for linear regression.

    Args:
        X (np.ndarray): Feature data.
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values (X @ w + b).

    Returns:
        tuple: A tuple containing:
            - dw (np.ndarray): Gradient with respect to weights.
            - db (float): Gradient with respect to bias.
    """
    n_samples = X.shape[0]
    errors = y_pred - y_true
    dw = (2 / n_samples) * X.T @ errors
    db = (2 / n_samples) * np.sum(errors)
    return dw, db

def predict(X, w, b):
    """
    Makes predictions using a linear model.

    Args:
        X (np.ndarray): Feature data.
        w (np.ndarray): Weights.
        b (float): Bias.

    Returns:
        np.ndarray: Predicted values.
    """
    return X @ w + b

if __name__ == '__main__':
    # Example usage (requires dummy data)
    X_dummy = np.array([[1], [2], [3]])
    y_true_dummy = np.array([[3], [5], [7]])
    w_dummy = np.array([[2]])
    b_dummy = 1
    y_pred_dummy = predict(X_dummy, w_dummy, b_dummy)
    print("Dummy Predictions:", y_pred_dummy)
    print("Dummy MSE:", mean_squared_error(y_true_dummy, y_pred_dummy))
    dw_dummy, db_dummy = mse_gradient(X_dummy, y_true_dummy, y_pred_dummy)
    print("Dummy Gradients (dw, db):", dw_dummy, db_dummy) 