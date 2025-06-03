#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for regularization examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, fetch_california_housing


def generate_synthetic_data(n_samples=100, n_features=20, noise=0.5, random_state=42, multicollinearity=False):
    """
    Generate synthetic data for regression with optional multicollinearity.
    
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
    multicollinearity : bool
        Whether to introduce high multicollinearity between features
        
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
    
    if multicollinearity:
        # Create base features
        n_base_features = int(n_features / 3)
        X_base = np.random.randn(n_samples, n_base_features)
        
        # Create features with high correlation
        X = np.zeros((n_samples, n_features))
        X[:, :n_base_features] = X_base
        
        # The rest of features are linear combinations of base features plus some noise
        for i in range(n_base_features, n_features):
            weights = np.random.uniform(-1, 1, n_base_features)
            X[:, i] = np.dot(X_base, weights) + np.random.normal(0, 0.1, n_samples)
        
        # Create sparse coefficients (many are zero)
        true_coef = np.zeros(n_features)
        active_indices = np.random.choice(n_features, size=int(n_features / 4), replace=False)
        true_coef[active_indices] = np.random.uniform(1, 5, len(active_indices))
        
        # Generate target with noise
        y = np.dot(X, true_coef) + np.random.normal(0, noise, n_samples)
    
    else:
        # Use scikit-learn's make_regression for simple case
        X, y, true_coef = make_regression(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=int(n_features * 0.8),  # Only some features are informative
            noise=noise,
            coef=True,  # Return true coefficients
            random_state=random_state
        )
    
    return X, y, true_coef


def load_real_data():
    """
    Load a real-world dataset suitable for demonstrating regularization.
    
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


def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to create a more complex dataset.
    
    Parameters:
    -----------
    X : ndarray
        Original feature matrix
    degree : int
        Polynomial degree
        
    Returns:
    --------
    X_poly : ndarray
        Feature matrix with polynomial features
    feature_names : list
        Names of all features including polynomial terms
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    n_samples, n_features = X.shape
    
    # Create original feature names if not provided
    base_feature_names = [f'X{i+1}' for i in range(n_features)]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Get new feature names
    feature_names = poly.get_feature_names_out(base_feature_names)
    
    return X_poly, feature_names


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for regression.
    
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


def analyze_multicollinearity(X, feature_names=None):
    """
    Analyze multicollinearity between features.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list or None
        List of feature names
        
    Returns:
    --------
    vif_df : DataFrame
        DataFrame containing VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    # Calculate VIF for each feature
    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X, i)
        vif_data.append({'Feature': feature_names[i], 'VIF': vif})
    
    # Create DataFrame and sort by VIF
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    return vif_df


def visualize_correlation_matrix(X, feature_names=None, title="Feature Correlation Matrix"):
    """
    Visualize correlation matrix for features.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list or None
        List of feature names
    title : str
        Plot title
    """
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f',
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # Identify highly correlated features
    high_corr_threshold = 0.7
    np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations
    high_corr_pairs = []
    
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > high_corr_threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    # Display high correlation pairs
    if high_corr_pairs:
        print(f"Feature pairs with correlation > {high_corr_threshold}:")
        for feature1, feature2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"{feature1} and {feature2}: {corr:.3f}")


def load_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data with multicollinearity:")
    X, y, true_coef = generate_synthetic_data(n_samples=200, n_features=10, multicollinearity=True)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    print(f"Number of non-zero true coefficients: {np.sum(true_coef != 0)}")
    
    # Analyze multicollinearity
    feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    vif_df = analyze_multicollinearity(X, feature_names)
    print("\nVariance Inflation Factors (VIF):")
    print(vif_df)
    
    # Visualize correlation matrix
    visualize_correlation_matrix(X, feature_names, "Synthetic Data Correlation Matrix")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"\nPreprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
