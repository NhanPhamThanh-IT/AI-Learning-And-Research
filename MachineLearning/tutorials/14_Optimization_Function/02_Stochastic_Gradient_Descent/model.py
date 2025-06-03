import numpy as np

def mean_squared_error_sgd(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) for a single sample.

    Args:
        y_true (float): Actual target value for a single sample.
        y_pred (float): Predicted target value for a single sample.

    Returns:
        float: The MSE for the single sample.
    """
    return (y_true - y_pred)**2

def mse_gradient_sgd(x, y_true, y_pred):
    """
    Calculates the gradient of the Mean Squared Error with respect to the weights and bias 
    for a single sample in linear regression.

    Args:
        x (np.ndarray): Feature data for a single sample (1, n_features).
        y_true (float): Actual target value for a single sample.
        y_pred (float): Predicted target value for a single sample.

    Returns:
        tuple: A tuple containing:
            - dw (np.ndarray): Gradient with respect to weights (n_features, 1).
            - db (float): Gradient with respect to bias.
    """
    error = y_pred - y_true
    dw = 2 * x.T * error
    db = 2 * error
    return dw, db

# Placeholder for a simple linear model prediction function
def predict(X, w, b):
    """
    Makes predictions using a linear model.

    Args:
        X (np.ndarray): Feature data (n_samples, n_features).
        w (np.ndarray): Weights (n_features, 1).
        b (float): Bias.

    Returns:
        np.ndarray: Predicted values (n_samples, 1).
    """
    return X @ w + b

if __name__ == '__main__':
    # Example usage (requires dummy data)
    x_dummy = np.array([[1]])
    y_true_dummy = 3
    w_dummy = np.array([[2]])
    b_dummy = 1
    y_pred_dummy = predict(x_dummy, w_dummy, b_dummy)[0, 0] # Get scalar prediction
    print("Dummy Prediction:", y_pred_dummy)
    print("Dummy MSE:", mean_squared_error_sgd(y_true_dummy, y_pred_dummy))
    dw_dummy, db_dummy = mse_gradient_sgd(x_dummy, y_true_dummy, y_pred_dummy)
    print("Dummy Gradients (dw, db):", dw_dummy, db_dummy) 