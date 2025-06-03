import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(n_samples=100):
    """Generates synthetic data for binary logistic regression."""
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 10
    y = (X[:, 0] + np.random.randn(n_samples) * 2.5 > 5).astype(int)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    
    # Optional: Visualize the generated data
    plt.scatter(X_train[y_train == 0], y_train[y_train == 0], label='Class 0')
    plt.scatter(X_train[y_train == 1], y_train[y_train == 1], label='Class 1')
    plt.xlabel("Feature")
    plt.ylabel("Class")
    plt.title("Generated Data for Logistic Regression")
    plt.legend()
    plt.show() 