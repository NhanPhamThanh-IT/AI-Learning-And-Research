import numpy as np

def generate_linear_data(n_samples=100, noise=5):
    """
    Generates synthetic linear data with added noise.

    Args:
        n_samples (int): Number of data points to generate.
        noise (float): The standard deviation of the Gaussian noise.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The feature data (n_samples, 1).
            - y (np.ndarray): The target data (n_samples, 1).
    """
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * noise
    return X, y

if __name__ == '__main__':
    X, y = generate_linear_data()
    print("Generated X shape:", X.shape)
    print("Generated y shape:", y.shape)
    print("First 5 X values:\n", X[:5])
    print("First 5 y values:\n", y[:5]) 