#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for logistic regression examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_breast_cancer


def generate_synthetic_data(n_samples=200, n_features=2, n_classes=2, n_informative=2,
                          random_state=42, class_sep=1.0, noise=0.1):
    """
    Generate synthetic data for logistic regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_classes : int
        Number of classes (2 for binary classification)
    n_informative : int
        Number of informative features
    random_state : int
        Random seed for reproducibility
    class_sep : float
        Class separation factor
    noise : float
        Noise level
        
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector (binary)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_informative,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=class_sep,
        flip_y=noise
    )
    
    return X, y


def load_real_data(dataset='breast_cancer'):
    """
    Load a real-world dataset for logistic regression.
    
    Parameters:
    -----------
    dataset : str
        Dataset name to load
        
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    feature_names : list
        List of feature names
    target_names : list
        List of target class names
    """
    if dataset == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")
    
    print(f"Loaded {dataset} dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Classes: {target_names}")
    
    return X, y, feature_names, target_names


def load_custom_data(filename, target_col=-1):
    """
    Load a custom dataset from a CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
    target_col : int
        Column index for the target variable
        
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    try:
        data = pd.read_csv(filename)
        
        # Extract column names
        columns = data.columns.tolist()
        target_column = columns[target_col if target_col >= 0 else len(columns) + target_col]
        feature_columns = [col for col in columns if col != target_column]
        
        # Extract X and y
        X = data[feature_columns].values
        y = data[target_column].values
        
        return X, y, feature_columns
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data instead...")
        X, y = generate_synthetic_data()
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for logistic regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    test_size : float
        Proportion of the dataset to include in the test split
    standardize : bool
        Whether to standardize features
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : ndarray
        Training features
    X_test : ndarray
        Testing features
    y_train : ndarray
        Training targets
    y_test : ndarray
        Testing targets
    scaler : StandardScaler or None
        Fitted scaler object if standardize=True
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y  # Ensure class distribution is preserved in train and test sets
    )
    
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_polynomial_features(X, degree=2):
    """
    Create polynomial features for logistic regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    degree : int
        Polynomial degree
        
    Returns:
    --------
    X_poly : ndarray
        Feature matrix with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    return X_poly


def visualize_data_2d(X, y, title="Data Visualization", feature_names=None):
    """
    Visualize 2D data for binary classification.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector
    title : str
        Plot title
    feature_names : list or None
        List of feature names
    """
    if X.shape[1] < 2:
        print("Error: Need at least 2 features for 2D visualization")
        return
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    plt.figure(figsize=(12, 10))
    
    # If more than 2 features, show pairwise plots for the first few features
    if X.shape[1] > 2:
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(X[:, :5], columns=feature_names[:5])
        df['target'] = y
        
        # Pairplot
        sns.pairplot(df, hue='target', palette='viridis')
        plt.suptitle(title, y=1.02)
        
    else:
        # Simple scatter plot for 2 features
        plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.7, marker='o')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.7, marker='^')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_decision_boundary(X, y, model, title="Decision Boundary", feature_names=None):
    """
    Visualize decision boundary for a binary classification model.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        Feature matrix (only works with 2D data)
    y : ndarray of shape (n_samples,)
        Target vector
    model : object
        Trained classification model with predict method
    title : str
        Plot title
    feature_names : list or None
        List of feature names
    """
    if X.shape[1] != 2:
        print("Error: Decision boundary visualization only works with 2D data")
        return
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    plt.figure(figsize=(10, 8))
    
    # Define the grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.7, marker='o', edgecolor='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.7, marker='^', edgecolor='k')
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(importances, feature_names=None, title="Feature Importance"):
    """
    Visualize feature importance or coefficients.
    
    Parameters:
    -----------
    importances : ndarray
        Feature importances or coefficients
    feature_names : list or None
        List of feature names
    title : str
        Plot title
    """
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(np.abs(importances))
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_importances)), sorted_importances)
    plt.yticks(range(len(sorted_importances)), sorted_features)
    plt.xlabel('Coefficient Magnitude')
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data for binary classification:")
    X, y = generate_synthetic_data(n_samples=200, n_features=2, class_sep=2.0)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Visualize the data
    visualize_data_2d(X, y, "Synthetic Binary Classification Data")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Testing class distribution: {np.bincount(y_test)}")
    
    # Load real data example
    X_real, y_real, feature_names_real, target_names_real = load_real_data('breast_cancer')
    print(f"Real data shape: X: {X_real.shape}, y: {y_real.shape}")
    
    # Preprocess real data
    X_train_real, X_test_real, y_train_real, y_test_real, _ = preprocess_data(X_real, y_real)
    print(f"Preprocessed real data shapes - X_train: {X_train_real.shape}, X_test: {X_test_real.shape}")
