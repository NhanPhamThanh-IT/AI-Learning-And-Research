import numpy as np

def create_dataset():
    """
    Creates a simple categorical dataset for demonstration.
    Similar to the golf example from the GfG article.
    Features: Outlook, Temperature, Humidity, Windy
    Label: Play Golf (Yes/No)
    """
    data = [
        ['Rainy', 'Hot', 'High', 'False', 'No'],
        ['Rainy', 'Hot', 'High', 'True', 'No'],
        ['Overcast', 'Hot', 'High', 'False', 'Yes'],
        ['Sunny', 'Mild', 'High', 'False', 'Yes'],
        ['Sunny', 'Cool', 'Normal', 'False', 'Yes'],
        ['Sunny', 'Cool', 'Normal', 'True', 'No'],
        ['Overcast', 'Cool', 'Normal', 'True', 'Yes'],
        ['Rainy', 'Mild', 'High', 'False', 'No'],
        ['Rainy', 'Cool', 'Normal', 'False', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'False', 'Yes'],
        ['Rainy', 'Mild', 'Normal', 'True', 'Yes'],
        ['Overcast', 'Mild', 'High', 'True', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'False', 'Yes'],
        ['Sunny', 'Mild', 'High', 'True', 'No']
    ]

    # Separate features (X) and labels (y)
    X = np.array([row[:-1] for row in data])
    y = np.array([row[-1] for row in data])

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
    X, y = create_dataset()
    print("Dataset created:")
    print("Features (X):\n", X)
    print("\nLabels (y):\n", y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nDataset split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
