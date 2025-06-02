#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polynomial regression model implementation from scratch and using scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt


class PolynomialRegressionFromScratch:
    """
    Polynomial Regression implementation from scratch.
    """
    
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, regularization=None, alpha=0.1):
        """
        Initialize the model.
        
        Parameters:
        -----------
        degree : int
            Polynomial degree
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        regularization : str or None
            Type of regularization ('l1', 'l2', or None)
        alpha : float
            Regularization strength
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.poly_features = None
    
    def _transform_features(self, X):
        """Transform features to polynomial features."""
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
            X_poly = self.poly_features.fit_transform(X)
        else:
            X_poly = self.poly_features.transform(X)
        return X_poly
    
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
        self : PolynomialRegressionFromScratch
            Fitted model
        """
        # Transform features to polynomial
        X_poly = self._transform_features(X)
        
        n_samples, n_features = X_poly.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X_poly, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_poly.T, (y_pred - y))
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
            cost = self._compute_cost(X_poly, y, y_pred)
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
        Predict using the polynomial model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X_poly = self._transform_features(X)
        return np.dot(X_poly, self.weights) + self.bias

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


class PolynomialRegressionModel:
    """
    A wrapper class providing both from-scratch and scikit-learn implementations
    of polynomial regression.
    """
    
    def __init__(self, degree=2, implementation='sklearn', **kwargs):
        """
        Initialize the polynomial regression model.
        
        Parameters:
        -----------
        degree : int
            Polynomial degree
        implementation : str
            Which implementation to use ('scratch' or 'sklearn')
        **kwargs : dict
            Additional parameters for the specific implementation
        """
        self.degree = degree
        self.implementation = implementation
        
        if implementation == 'scratch':
            self.model = PolynomialRegressionFromScratch(degree=degree, **kwargs)
        else:  # sklearn
            self.model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=True),
                SklearnLinearRegression(**kwargs)
            )
    
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
        mae = mean_absolute_error(y, y_pred)
        
        if self.implementation == 'scratch':
            metrics = {
                'degree': self.degree,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'weights': self.model.weights,
                'bias': self.model.bias,
                'cost_history': self.model.cost_history
            }
        else:  # sklearn
            # Extract coefficients from the pipeline
            linear_model = self.model.steps[-1][1]
            poly_features = self.model.steps[0][1]
            
            metrics = {
                'degree': self.degree,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'coef': linear_model.coef_,
                'intercept': linear_model.intercept_,
                'feature_names': poly_features.get_feature_names_out()
            }
        
        return metrics


def find_best_degree(X, y, max_degree=10, cv=5, plot=True):
    """
    Find the best polynomial degree using cross-validation.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    max_degree : int
        Maximum polynomial degree to test
    cv : int
        Number of cross-validation folds
    plot : bool
        Whether to plot the validation curves
        
    Returns:
    --------
    best_degree : int
        Best polynomial degree
    cv_scores : dict
        Cross-validation scores for each degree
    """
    degrees = range(1, max_degree + 1)
    train_scores = []
    val_scores = []
    
    for degree in degrees:
        # Create model
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            SklearnLinearRegression()
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        train_score = -np.mean(cv_scores)  # Negative MSE to positive MSE
        val_score = -np.mean(cv_scores)  # Same for validation
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    # Find best degree
    best_degree = degrees[np.argmin(val_scores)]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(degrees, train_scores, 'o-', label='Training MSE')
        plt.plot(degrees, val_scores, 'o-', label='Validation MSE')
        plt.axvline(x=best_degree, color='r', linestyle='--', label=f'Best degree: {best_degree}')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error')
        plt.title('Validation Curve for Polynomial Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    cv_scores = {
        'degrees': list(degrees),
        'train_scores': train_scores,
        'val_scores': val_scores,
        'best_degree': best_degree
    }
    
    return best_degree, cv_scores


def plot_learning_curves(X, y, degrees=[1, 3, 5], cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves for different polynomial degrees.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    degrees : list
        List of polynomial degrees to test
    cv : int
        Number of cross-validation folds
    train_sizes : ndarray
        Training set sizes to plot the learning curves
    """
    from sklearn.model_selection import learning_curve
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # Create model
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            SklearnLinearRegression()
        )
        
        # Calculate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, 
            train_sizes=train_sizes,
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Convert negative MSE to positive MSE
        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        train_scores_std = np.std(-train_scores, axis=1)
        val_scores_std = np.std(-val_scores, axis=1)
        
        # Plot learning curve
        plt.subplot(1, len(degrees), i + 1)
        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='b')
        plt.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='b', label='Training MSE')
        plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='g', label='Validation MSE')
        plt.title(f'Degree {degree} - Learning Curve')
        plt.xlabel('Training set size')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from data import generate_synthetic_data, preprocess_data, visualize_data_and_fit
    
    # Generate synthetic data for polynomial fitting
    print("Generating synthetic polynomial data:")
    X, y, true_func = generate_synthetic_data(n_samples=100, function_type='polynomial', noise=0.5)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y, standardize=False)
    
    # Find best degree
    print("\nFinding best polynomial degree:")
    best_degree, cv_scores = find_best_degree(X_train, y_train, max_degree=8, cv=5, plot=True)
    print(f"Best degree: {best_degree}")
    
    # Train model with best degree
    print(f"\nTraining polynomial regression with degree {best_degree}:")
    model = PolynomialRegressionModel(degree=best_degree)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print("Test metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Visualize the fit
    visualize_data_and_fit(X, y, model, f"Polynomial Regression (Degree {best_degree})")
    
    # Show learning curves for different degrees
    print("\nComparing learning curves for different degrees:")
    plot_learning_curves(X, y, degrees=[1, best_degree, 8], cv=5)
