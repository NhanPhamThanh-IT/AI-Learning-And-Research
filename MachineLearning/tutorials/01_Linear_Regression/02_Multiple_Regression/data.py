#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for multiple linear regression examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, make_regression


def generate_synthetic_data(n_samples=200, n_features=3, noise=0.5, random_state=42):
    """
    Generate synthetic data for multiple linear regression.
    
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
    X, y, coef = make_regression(n_samples=n_samples, 
                                n_features=n_features,
                                noise=noise,
                                coef=True,
                                random_state=random_state)
    
    return X, y, coef


def load_real_data():
    """
    Load a real-world dataset for multiple regression.
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    # Load California housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target
    feature_names = california.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    return X, y, feature_names


def load_custom_data(filename):
    """
    Load a custom dataset from a CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
        
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
        feature_names = data.columns[:-1].tolist()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y, feature_names
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data instead...")
        X, y, _ = generate_synthetic_data()
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for multiple linear regression.
    
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


def analyze_features(X, y, feature_names):
    """
    Analyze features for multiple regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Statistical summary
    print("Statistical Summary:")
    print(df.describe())
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Pairplot for the first few features (if there are many)
    n_features = min(5, len(feature_names))
    selected_cols = feature_names[:n_features] + ['target']
    plt.figure(figsize=(15, 10))
    sns.pairplot(df[selected_cols])
    plt.suptitle('Feature Pairplot', y=1.02)
    plt.show()
    
    # Check for multicollinearity
    print("\nChecking for multicollinearity:")
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_names
    
    # Calculate VIF for each feature
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create a copy of X as a DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    print(vif_data)


if __name__ == "__main__":
    # Example usage
    X, y, true_coef = generate_synthetic_data(n_samples=200, n_features=5, noise=0.5)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    print(f"True coefficients: {true_coef}")
    
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    analyze_features(X, y, feature_names)
