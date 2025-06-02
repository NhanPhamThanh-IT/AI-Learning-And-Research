#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for linear regression examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_samples=100, n_features=1, noise=0.3, random_state=42):
    """
    Generate synthetic data for linear regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features (independent variables)
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector
    true_coef : ndarray of shape (n_features,)
        True coefficients used to generate the data
    """
    np.random.seed(random_state)
    
    # Generate random coefficients
    true_coef = np.random.randn(n_features)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with noise
    y = X.dot(true_coef) + np.random.normal(0, noise, n_samples)
    
    return X, y, true_coef


def load_sample_data(filename=None):
    """
    Load sample dataset for linear regression.
    If no filename is provided, generates synthetic data.
    
    Parameters:
    -----------
    filename : str, optional
        Path to CSV file containing the dataset
        
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    """
    if filename is None:
        X, y, _ = generate_synthetic_data()
        return X, y
    
    try:
        data = pd.read_csv(filename)
        X = data.iloc[:, :-1].values  # All columns except the last one
        y = data.iloc[:, -1].values   # Last column as target
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data instead...")
        X, y, _ = generate_synthetic_data()
        return X, y


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for linear regression.
    
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
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def visualize_data(X, y, title="Data Visualization"):
    """
    Visualize data for linear regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    if X.shape[1] == 1:
        # For simple linear regression (1D)
        plt.scatter(X, y, alpha=0.7)
        plt.xlabel('X')
        plt.ylabel('y')
    else:
        # For multiple linear regression, create a pair plot of first 3 features
        n_features = min(3, X.shape[1])
        fig, axes = plt.subplots(1, n_features, figsize=(15, 5))
        
        for i in range(n_features):
            if n_features == 1:
                ax = axes
            else:
                ax = axes[i]
            
            ax.scatter(X[:, i], y, alpha=0.7)
            ax.set_xlabel(f'Feature {i+1}')
            if i == 0:
                ax.set_ylabel('Target')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    X, y, true_coef = generate_synthetic_data(n_samples=200, n_features=3, noise=0.5)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    print(f"True coefficients: {true_coef}")
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    visualize_data(X[:, :1], y, "Sample Synthetic Data")