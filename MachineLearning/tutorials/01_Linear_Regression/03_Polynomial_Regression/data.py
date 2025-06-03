#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for polynomial regression examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def generate_synthetic_data(n_samples=100, noise=0.3, function_type='polynomial', random_state=42):
    """
    Generate synthetic data for polynomial regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    function_type : str
        Type of function to generate data from ('polynomial', 'sine', 'exponential')
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : ndarray of shape (n_samples, 1)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector
    true_function : function
        True function used to generate the data
    """
    np.random.seed(random_state)
    
    # Generate random X values between -3 and 3
    X = np.random.uniform(-3, 3, (n_samples, 1))
    
    # Sort X for easier plotting
    X = np.sort(X, axis=0)
    
    # Generate target based on function type
    if function_type == 'polynomial':
        # y = 1 + 2x + 3x^2 - 2x^3 + noise
        y = 1 + 2*X.ravel() + 3*X.ravel()**2 - 2*X.ravel()**3
        true_function = lambda x: 1 + 2*x + 3*x**2 - 2*x**3
    
    elif function_type == 'sine':
        # y = sin(2*pi*x) + noise
        y = np.sin(2 * np.pi * X.ravel())
        true_function = lambda x: np.sin(2 * np.pi * x)
    
    elif function_type == 'exponential':
        # y = 0.5 * exp(x) + noise
        y = 0.5 * np.exp(X.ravel())
        true_function = lambda x: 0.5 * np.exp(x)
    
    else:
        raise ValueError("Invalid function_type. Choose 'polynomial', 'sine', or 'exponential'.")
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    return X, y, true_function


def load_real_data(dataset='boston'):
    """
    Load a real-world dataset for polynomial regression.
    
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
    feature_name : str
        Name of the selected feature
    """
    if dataset == 'boston':
        from sklearn.datasets import load_boston
        try:
            boston = load_boston()
            
            # Select a single feature for polynomial regression
            # LSTAT (% lower status of the population) tends to have non-linear relationship with price
            feature_idx = 12  # LSTAT index
            X = boston.data[:, [feature_idx]]
            y = boston.target
            feature_name = boston.feature_names[feature_idx]
            
        except:
            print("Boston dataset is no longer available in scikit-learn. Using synthetic data instead.")
            X, y, _ = generate_synthetic_data(n_samples=506)  # boston dataset size
            feature_name = "Synthetic_X"
    
    elif dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        
        # Select a single feature
        feature_idx = 2  # bmi
        X = diabetes.data[:, [feature_idx]]
        y = diabetes.target
        feature_name = diabetes.feature_names[feature_idx]
    
    else:
        raise ValueError("Invalid dataset name. Choose 'boston' or 'diabetes'.")
    
    return X, y, feature_name


def load_custom_data(filename, x_col=0, y_col=-1):
    """
    Load a custom dataset from a CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
    x_col : int
        Column index for X variable
    y_col : int
        Column index for y variable
        
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    feature_name : str
        Name of the selected feature
    """
    try:
        data = pd.read_csv(filename)
        
        # Extract column names
        columns = data.columns.tolist()
        x_column = columns[x_col]
        y_column = columns[y_col if y_col >= 0 else len(columns) + y_col]
        
        # Extract X and y
        X = data.iloc[:, [x_col]].values
        y = data.iloc[:, y_col].values
        
        return X, y, x_column
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data instead...")
        X, y, _ = generate_synthetic_data()
        return X, y, "Synthetic_X"


def create_polynomial_features(X, degree=2, include_bias=True):
    """
    Create polynomial features from input matrix.
    
    Parameters:
    -----------
    X : ndarray
        Input feature matrix
    degree : int
        Maximum polynomial degree
    include_bias : bool
        Whether to include the bias term (intercept)
        
    Returns:
    --------
    X_poly : ndarray
        Feature matrix with polynomial features
    poly_features : PolynomialFeatures
        Fitted PolynomialFeatures transformer
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly_features.fit_transform(X)
    
    return X_poly, poly_features


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for polynomial regression.
    
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


def visualize_data_and_fit(X, y, model=None, title="Data and Model Fit", x_label="X", y_label="y"):
    """
    Visualize data and model fit for polynomial regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    model : object or None
        Fitted model with predict method
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of data
    plt.scatter(X, y, alpha=0.7, label='Data points')
    
    # If model is provided, plot the fit
    if model is not None:
        # Generate more points for smooth curve
        X_curve = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        # Transform if this is original X rather than polynomial
        y_pred = model.predict(X_curve)
        plt.plot(X_curve, y_pred, 'r-', label='Model fit', linewidth=2)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_polynomial_fits(X, y, models, degrees, x_label="X", y_label="y"):
    """
    Visualize fits of polynomial models with different degrees.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    models : list
        List of fitted models
    degrees : list
        List of degrees corresponding to models
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    """
    plt.figure(figsize=(15, 10))
    n_models = len(models)
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    for i, (model, degree) in enumerate(zip(models, degrees)):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Scatter plot of data
        plt.scatter(X, y, alpha=0.7, label='Data')
        
        # Generate smooth curve for model prediction
        X_curve = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        y_pred = model.predict(X_curve)
        
        plt.plot(X_curve, y_pred, 'r-', label=f'Degree {degree}', linewidth=2)
        plt.title(f'Polynomial Degree {degree}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data:")
    X, y, true_func = generate_synthetic_data(n_samples=100, function_type='polynomial', noise=0.5)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    
    # Visualize the data
    visualize_data_and_fit(X, y, None, "Synthetic Data")
    
    # Create polynomial features
    X_poly, poly_features = create_polynomial_features(X, degree=3)
    print(f"Polynomial features shape: {X_poly.shape}")
    print(f"Feature names: {poly_features.get_feature_names_out()}")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, standardize=False)
    print(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
