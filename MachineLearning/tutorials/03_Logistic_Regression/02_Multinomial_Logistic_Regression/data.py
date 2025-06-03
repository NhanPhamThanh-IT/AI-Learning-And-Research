#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation, loading, and preprocessing functions for multinomial logistic regression examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, load_iris, load_digits


def generate_synthetic_data(n_samples=300, n_features=2, n_classes=3, n_informative=2,
                          random_state=42, class_sep=1.0, noise=0.1):
    """
    Generate synthetic data for multinomial logistic regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_classes : int
        Number of classes
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
        Target vector (multi-class)
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


def load_real_data(dataset='iris'):
    """
    Load a real-world dataset for multinomial logistic regression.
    
    Parameters:
    -----------
    dataset : str
        Dataset name to load ('iris' or 'digits')
        
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
    if dataset == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
    
    elif dataset == 'digits':
        data = load_digits()
        X = data.data
        y = data.target
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
    
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
    target_names : list
        List of target class names
    """
    try:
        data = pd.read_csv(filename)
        
        # Extract column names
        columns = data.columns.tolist()
        target_column = columns[target_col if target_col >= 0 else len(columns) + target_col]
        feature_columns = [col for col in columns if col != target_column]
        
        # Extract X and y
        X = data[feature_columns].values
        y_raw = data[target_column].values
        
        # Encode target if it's not numeric
        if not np.issubdtype(y_raw.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)
            target_names = label_encoder.classes_
        else:
            y = y_raw
            target_names = [str(i) for i in np.unique(y)]
        
        return X, y, feature_columns, target_names
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data instead...")
        X, y = generate_synthetic_data(n_classes=3)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        target_names = [f"Class_{i}" for i in range(3)]
        return X, y, feature_names, target_names


def preprocess_data(X, y, test_size=0.2, standardize=True, random_state=42):
    """
    Preprocess data for multinomial logistic regression.
    
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
    Create polynomial features for multinomial logistic regression.
    
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


def visualize_data_2d(X, y, title="Data Visualization", target_names=None):
    """
    Visualize 2D data for multi-class classification.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector
    title : str
        Plot title
    target_names : list or None
        List of target class names
    """
    if X.shape[1] < 2:
        print("Error: Need at least 2 features for 2D visualization")
        return
    
    # Create class labels for the plot
    if target_names is None:
        target_names = [f"Class {i}" for i in range(len(np.unique(y)))]
    
    plt.figure(figsize=(12, 10))
    
    # If more than 2 features, show pairwise plots for the first few features
    if X.shape[1] > 2:
        # Create a DataFrame for easier plotting
        feature_names = [f"Feature {i+1}" for i in range(min(5, X.shape[1]))]
        df = pd.DataFrame(X[:, :5], columns=feature_names)
        df['target'] = y
        
        # Convert numeric target to string labels
        df['target'] = df['target'].apply(lambda x: target_names[x])
        
        # Pairplot
        sns.pairplot(df, hue='target', palette='viridis')
        plt.suptitle(title, y=1.02)
        
    else:
        # Simple scatter plot for 2 features
        unique_classes = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            plt.scatter(
                X[y == cls, 0], 
                X[y == cls, 1], 
                label=target_names[i], 
                alpha=0.7, 
                color=colors[i]
            )
            
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_decision_boundaries(X, y, model, title="Decision Boundaries", target_names=None):
    """
    Visualize decision boundaries for a multi-class classification model.
    
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
    target_names : list or None
        List of target class names
    """
    if X.shape[1] != 2:
        print("Error: Decision boundary visualization only works with 2D data")
        return
    
    # Create class labels for the plot
    if target_names is None:
        target_names = [f"Class {i}" for i in range(len(np.unique(y)))]
    
    plt.figure(figsize=(10, 8))
    
    # Define the grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the data points
    unique_classes = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    
    for i, cls in enumerate(unique_classes):
        plt.scatter(
            X[y == cls, 0], 
            X[y == cls, 1], 
            label=target_names[i], 
            edgecolor='k',
            alpha=0.8, 
            color=colors[i]
        )
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(importances, feature_names=None, title="Feature Importance"):
    """
    Visualize feature importance or coefficients for each class.
    
    Parameters:
    -----------
    importances : ndarray of shape (n_classes, n_features)
        Feature importances or coefficients for each class
    feature_names : list or None
        List of feature names
    title : str
        Plot title
    """
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(importances.shape[1])]
    
    n_classes = importances.shape[0]
    
    plt.figure(figsize=(12, n_classes * 3))
    
    for i in range(n_classes):
        plt.subplot(n_classes, 1, i + 1)
        
        # Sort features by importance
        indices = np.argsort(np.abs(importances[i]))
        sorted_importances = importances[i][indices]
        sorted_features = [feature_names[j] for j in indices]
        
        plt.barh(range(len(sorted_importances)), sorted_importances)
        plt.yticks(range(len(sorted_importances)), sorted_features)
        plt.xlabel(f'Coefficient Magnitude (Class {i})')
        plt.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.show()


def visualize_digits_dataset():
    """
    Load and visualize the digits dataset.
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Show some example digits
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show distribution of classes
    plt.figure(figsize=(10, 5))
    class_counts = np.bincount(y)
    plt.bar(range(10), class_counts)
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Distribution of Digits')
    plt.xticks(range(10))
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()
    
    return X, y


def load_data():
    X, y = make_classification(n_samples=200, n_features=8, n_classes=3, n_informative=6, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data for multi-class classification:")
    X, y = generate_synthetic_data(n_samples=300, n_features=2, n_classes=3, class_sep=1.5)
    print(f"Generated data with shape X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Visualize the data
    visualize_data_2d(X, y, "Synthetic Multi-class Classification Data")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Testing class distribution: {np.bincount(y_test)}")
    
    # Load real data example
    print("\nLoading Iris dataset:")
    X_iris, y_iris, feature_names_iris, target_names_iris = load_real_data('iris')
    visualize_data_2d(X_iris, y_iris, "Iris Dataset", target_names_iris)
    
    # Load and visualize digits dataset
    print("\nLoading and visualizing Digits dataset:")
    X_digits, y_digits = visualize_digits_dataset()
