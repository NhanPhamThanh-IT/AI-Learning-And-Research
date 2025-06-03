import numpy as np

def create_synthetic_dataset(n_samples=100):
    """
    Creates a simple synthetic dataset with continuous features for demonstration.
    """
    np.random.seed(42) # for reproducibility

    # Class 0: Mean = [2, 2], Covariance = [[0.5, 0], [0, 0.5]]
    X0 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples // 2)
    y0 = np.zeros(n_samples // 2)

    # Class 1: Mean = [5, 5], Covariance = [[0.7, 0], [0, 0.7]]
    X1 = np.random.multivariate_normal([5, 5], [[0.7, 0], [0, 0.7]], n_samples // 2)
    y1 = np.ones(n_samples // 2)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

def split_data(X, y, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets.
    """
    np.random.seed(42) # for reproducibility
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_size = int(X.shape[0] * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X, y = create_synthetic_dataset()
    print("Dataset created:")
    print("Features (X) shape:", X.shape)
    print("Labels (y) shape:", y.shape)
    print("\nFirst 5 samples:")
    for i in range(5):
        print(f"X: {X[i]}, y: {int(y[i])}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nDataset split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
