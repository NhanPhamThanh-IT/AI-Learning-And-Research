#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running linear regression analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import generate_synthetic_data, preprocess_data, visualize_data
from model import LinearRegressionModel


def plot_predictions(X, y, y_pred, title="Model Predictions"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    
    if X.shape[1] == 1:
        # For simple linear regression
        plt.scatter(X, y, color='blue', alpha=0.7, label='Actual')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
        plt.xlabel('X')
        plt.ylabel('y')
    else:
        # For multiple features, plot actual vs predicted
        plt.scatter(y, y_pred, alpha=0.7)
        
        # Plot perfect prediction line
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_residuals(y, y_pred, title="Residuals Analysis"):
    """Plot residuals."""
    residuals = y - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Residuals vs predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Residuals distribution
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(cost_history, title="Learning Curve"):
    """Plot the learning curve (cost history)."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_linear_regression_analysis(n_samples=100, n_features=1, noise=0.3, 
                                 test_size=0.2, standardize=True,
                                 implementation='sklearn', **model_params):
    """
    Run a complete linear regression analysis pipeline.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features (independent variables)
    noise : float
        Standard deviation of Gaussian noise
    test_size : float
        Proportion of the dataset to include in the test split
    standardize : bool
        Whether to standardize features
    implementation : str
        Which implementation to use ('scratch' or 'sklearn')
    **model_params : dict
        Additional parameters for the model
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Running Linear Regression Analysis ({implementation} implementation)")
    print(f"{'='*50}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y, true_coef = generate_synthetic_data(n_samples, n_features, noise)
    print(f"Data shape: X: {X.shape}, y: {y.shape}")
    print(f"True coefficients: {true_coef}")
    
    # Visualize the data
    print("\nVisualizing raw data...")
    if n_features <= 3:
        visualize_data(X, y, "Raw Data")
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, standardize=standardize
    )
    print(f"Train shapes: X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test shapes: X: {X_test.shape}, y: {y_test.shape}")
    
    # Create and fit model
    print(f"\nTraining {implementation} model...")
    model = LinearRegressionModel(implementation, **model_params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    if implementation == 'scratch':
        print(f"Learned weights: {metrics['weights']}")
        print(f"Learned bias: {metrics['bias']}")
    else:
        print(f"Learned coefficients: {metrics['coef']}")
        print(f"Learned intercept: {metrics['intercept']}")
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Plot predictions
    print("\nPlotting results...")
    if n_features == 1:
        plot_predictions(X_test, y_test, y_pred_test, "Test Set Predictions")
    else:
        plot_predictions(X_train, y_train, y_pred_train, "Training Set: Actual vs Predicted")
        plot_predictions(X_test, y_test, y_pred_test, "Test Set: Actual vs Predicted")
    
    # Plot residuals
    plot_residuals(y_test, y_pred_test, "Residuals Analysis (Test Set)")
    
    # Plot learning curve if available
    if implementation == 'scratch' and 'cost_history' in metrics:
        plot_learning_curve(metrics['cost_history'], "Gradient Descent Learning Curve")
    
    print(f"\n{'='*50}")
    print("Analysis Complete!")
    print(f"{'='*50}")
    
    return metrics


if __name__ == "__main__":
    # Run simple linear regression with sklearn
    run_linear_regression_analysis(
        n_samples=100, 
        n_features=1, 
        noise=0.5, 
        implementation='sklearn'
    )
    
    # Run multiple linear regression with from-scratch implementation
    run_linear_regression_analysis(
        n_samples=200, 
        n_features=3, 
        noise=0.5, 
        implementation='scratch',
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        alpha=0.1
    )