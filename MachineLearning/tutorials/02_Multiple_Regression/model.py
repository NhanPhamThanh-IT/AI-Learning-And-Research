#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiple linear regression model implementation from scratch and using scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultipleLinearRegressionFromScratch:
    """
    Multiple Linear Regression implementation from scratch using gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, alpha=0.1):
        """
        Initialize the model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        regularization : str or None
            Type of regularization ('l1', 'l2', or None)
        alpha : float
            Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Fit the model to the data using gradient descent.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
        y : ndarray of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        self : MultipleLinearRegressionFromScratch
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Add regularization to gradients if specified
            if self.regularization == 'l2':  # Ridge
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == 'l1':  # Lasso
                dw += (self.alpha / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
        
        return self
    
    def _compute_cost(self, X, y, y_pred):
        """Compute the cost function (MSE with optional regularization)."""
        n_samples = X.shape[0]
        mse = np.mean((y_pred - y) ** 2)
        
        if self.regularization == 'l2':  # Ridge
            reg_term = (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
            return mse + reg_term
        elif self.regularization == 'l1':  # Lasso
            reg_term = (self.alpha / n_samples) * np.sum(np.abs(self.weights))
            return mse + reg_term
        else:
            return mse
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Return the coefficient of determination (R^2) of the prediction.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True values for test samples
            
        Returns:
        --------
        score : float
            R^2 score
        """
        y_pred = self.predict(X)
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    def get_coef_stats(self, X, y, feature_names=None):
        """
        Get statistics about coefficients.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
        y : ndarray of shape (n_samples,)
            Target vector
        feature_names : list, optional
            List of feature names
            
        Returns:
        --------
        coef_stats : DataFrame
            DataFrame containing coefficient statistics
        """
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(len(self.weights))]
            
        # Calculate standard errors and t-statistics
        n_samples = X.shape[0]
        n_features = X.shape[1]
        y_pred = self.predict(X)
        residuals = y - y_pred
        residual_sum_sq = np.sum(residuals**2)
        sigma_squared = residual_sum_sq / (n_samples - n_features - 1)
        
        # Calculate the variance-covariance matrix
        X_with_intercept = np.column_stack((np.ones(n_samples), X))
        cov_matrix = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)) * sigma_squared
        
        # Extract standard errors
        std_errors = np.sqrt(np.diag(cov_matrix))
        intercept_se = std_errors[0]
        coef_se = std_errors[1:]
        
        # Calculate t-statistics
        t_stat_intercept = self.bias / intercept_se
        t_stats = self.weights / coef_se
        
        # Create DataFrame with coefficient statistics
        coef_stats = pd.DataFrame({
            'feature': ['intercept'] + feature_names,
            'coefficient': [self.bias] + list(self.weights),
            'std_error': [intercept_se] + list(coef_se),
            't_statistic': [t_stat_intercept] + list(t_stats)
        })
        
        return coef_stats


class MultipleRegressionModel:
    """
    A wrapper class providing both from-scratch and scikit-learn implementations
    of multiple linear regression.
    """
    
    def __init__(self, implementation='sklearn', **kwargs):
        """
        Initialize the multiple linear regression model.
        
        Parameters:
        -----------
        implementation : str
            Which implementation to use ('scratch' or 'sklearn')
        **kwargs : dict
            Additional parameters for the specific implementation
        """
        self.implementation = implementation
        
        if implementation == 'scratch':
            self.model = MultipleLinearRegressionFromScratch(**kwargs)
        else:  # sklearn
            self.model = SklearnLinearRegression(**kwargs)
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)
    
    def evaluate(self, X, y, feature_names=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : ndarray
            Test features
        y : ndarray
            True test targets
        feature_names : list, optional
            List of feature names
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate adjusted R-squared
        n_samples = X.shape[0]
        n_features = X.shape[1]
        adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        
        if self.implementation == 'scratch':
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'adjusted_r2': adjusted_r2,
                'mae': mae,
                'weights': self.model.weights,
                'bias': self.model.bias,
                'cost_history': self.model.cost_history
            }
            
            # Add coefficient statistics if feature_names are provided
            if feature_names is not None:
                coef_stats = self.model.get_coef_stats(X, y, feature_names)
                metrics['coef_stats'] = coef_stats
                
        else:  # sklearn
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'adjusted_r2': adjusted_r2,
                'mae': mae,
                'coef': self.model.coef_,
                'intercept': self.model.intercept_
            }
            
        return metrics
    
    def plot_residuals(self, X, y):
        """
        Plot residual analysis for the multiple regression model.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            True target values
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Residuals vs Fitted values
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='-')
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        
        # Plot 2: Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='-')
        axes[1].set_xlabel('Residual value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Histogram of Residuals')
        
        # Plot 3: Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, plot=axes[2])
        axes[2].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print residual statistics
        print("Residual Statistics:")
        print(f"Mean of residuals: {np.mean(residuals):.4f}")
        print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
        print(f"Residual range: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")


if __name__ == "__main__":
    # Example usage
    from data import generate_synthetic_data, preprocess_data
    
    # Generate synthetic data with multiple features
    X, y, true_coef = generate_synthetic_data(n_samples=200, n_features=5, noise=0.5)
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # Train and evaluate models
    print("\nTraining scratch implementation:")
    scratch_model = MultipleRegressionModel('scratch', learning_rate=0.01, n_iterations=2000)
    scratch_model.fit(X_train, y_train)
    scratch_metrics = scratch_model.evaluate(X_test, y_test, feature_names)
    print(f"Scratch model metrics: {scratch_metrics}")
    
    print("\nTraining sklearn implementation:")
    sklearn_model = MultipleRegressionModel('sklearn')
    sklearn_model.fit(X_train, y_train)
    sklearn_metrics = sklearn_model.evaluate(X_test, y_test)
    print(f"Sklearn model metrics: {sklearn_metrics}")
    
    print("\nComparing coefficients:")
    print(f"True coefficients: {true_coef}")
    print(f"Scratch model coefficients: {scratch_model.model.weights}")
    print(f"Sklearn model coefficients: {sklearn_model.model.coef_}")
    
    # Plot residuals for sklearn model
    sklearn_model.plot_residuals(X_test, y_test)
