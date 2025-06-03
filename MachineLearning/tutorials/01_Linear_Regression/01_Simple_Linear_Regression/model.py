#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear regression model implementation from scratch and using scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionFromScratch:
    """
    Linear Regression implementation from scratch using gradient descent.
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
        
    def _add_intercept(self, X):
        """Add intercept term to features."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
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
        self : LinearRegressionFromScratch
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


class LinearRegressionModel:
    """
    A wrapper class providing both from-scratch and scikit-learn implementations
    of linear regression.
    """
    
    def __init__(self, implementation='sklearn', **kwargs):
        """
        Initialize the linear regression model.
        
        Parameters:
        -----------
        implementation : str
            Which implementation to use ('scratch' or 'sklearn')
        **kwargs : dict
            Additional parameters for the specific implementation
        """
        self.implementation = implementation
        
        if implementation == 'scratch':
            self.model = LinearRegressionFromScratch(**kwargs)
        else:  # sklearn
            self.model = SklearnLinearRegression(**kwargs)
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : ndarray
            Test features
        y : ndarray
            True test targets
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        if self.implementation == 'scratch' and hasattr(self.model, 'weights'):
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'weights': self.model.weights,
                'bias': self.model.bias,
                'cost_history': self.model.cost_history
            }
        else:
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'coef': self.model.coef_,
                'intercept': self.model.intercept_
            }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    from data import generate_synthetic_data, preprocess_data
    
    # Generate synthetic data
    X, y, true_coef = generate_synthetic_data(n_samples=100, n_features=1)
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # From-scratch implementation
    scratch_model = LinearRegressionModel('scratch', learning_rate=0.01, n_iterations=1000)
    scratch_model.fit(X_train, y_train)
    scratch_metrics = scratch_model.evaluate(X_test, y_test)
    print(f"Scratch model metrics: {scratch_metrics}")
    
    # Scikit-learn implementation
    sklearn_model = LinearRegressionModel('sklearn')
    sklearn_model.fit(X_train, y_train)
    sklearn_metrics = sklearn_model.evaluate(X_test, y_test)
    print(f"Sklearn model metrics: {sklearn_metrics}")