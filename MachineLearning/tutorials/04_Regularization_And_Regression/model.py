#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regularized regression model implementations from scratch and using scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        return self.model.score(X, y)


class LassoRegression:
    def __init__(self, alpha=0.1):
        self.model = Lasso(alpha=alpha)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        return self.model.score(X, y)


class RidgeRegressionFromScratch:
    """
    Ridge Regression (L2 regularization) implementation from scratch.
    """
    
    def __init__(self, alpha=1.0, n_iterations=1000, learning_rate=0.01):
        """
        Initialize the model.
        
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
        n_iterations : int
            Number of iterations for gradient descent
        learning_rate : float
            Step size for gradient descent
        """
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
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
        self : RidgeRegressionFromScratch
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients with L2 regularization
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
        
        return self
    
    def _compute_cost(self, X, y, y_pred):
        """Compute the cost function (MSE with L2 regularization)."""
        n_samples = X.shape[0]
        mse = np.mean((y_pred - y) ** 2)
        l2_reg = (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
        return mse + l2_reg
    
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


class LassoRegressionFromScratch:
    """
    Lasso Regression (L1 regularization) implementation from scratch.
    """
    
    def __init__(self, alpha=1.0, n_iterations=1000, learning_rate=0.01):
        """
        Initialize the model.
        
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
        n_iterations : int
            Number of iterations for gradient descent
        learning_rate : float
            Step size for gradient descent
        """
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
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
        self : LassoRegressionFromScratch
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients with L1 regularization
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * np.sign(self.weights))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
        
        return self
    
    def _compute_cost(self, X, y, y_pred):
        """Compute the cost function (MSE with L1 regularization)."""
        n_samples = X.shape[0]
        mse = np.mean((y_pred - y) ** 2)
        l1_reg = (self.alpha / n_samples) * np.sum(np.abs(self.weights))
        return mse + l1_reg
    
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


class ElasticNetFromScratch:
    """
    Elastic Net (L1 + L2 regularization) implementation from scratch.
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, n_iterations=1000, learning_rate=0.01):
        """
        Initialize the model.
        
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
        l1_ratio : float
            Ratio of L1 regularization (0 to 1)
            l1_ratio=0 is Ridge, l1_ratio=1 is Lasso
        n_iterations : int
            Number of iterations for gradient descent
        learning_rate : float
            Step size for gradient descent
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
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
        self : ElasticNetFromScratch
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients with combined L1 and L2 regularization
            l1_term = self.alpha * self.l1_ratio * np.sign(self.weights)
            l2_term = self.alpha * (1 - self.l1_ratio) * self.weights
            
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)) + l1_term + l2_term)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
        
        return self
    
    def _compute_cost(self, X, y, y_pred):
        """Compute the cost function (MSE with L1 and L2 regularization)."""
        n_samples = X.shape[0]
        mse = np.mean((y_pred - y) ** 2)
        l1_reg = (self.alpha * self.l1_ratio / n_samples) * np.sum(np.abs(self.weights))
        l2_reg = (self.alpha * (1 - self.l1_ratio) / (2 * n_samples)) * np.sum(self.weights ** 2)
        return mse + l1_reg + l2_reg
    
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


class RegularizedRegressionModel:
    """
    A wrapper class for regularized regression models.
    """
    
    def __init__(self, model_type='ridge', implementation='sklearn', **kwargs):
        """
        Initialize the regularized regression model.
        
        Parameters:
        -----------
        model_type : str
            Type of regularization ('ridge', 'lasso', or 'elasticnet')
        implementation : str
            Which implementation to use ('scratch' or 'sklearn')
        **kwargs : dict
            Additional parameters for the specific model
        """
        self.model_type = model_type
        self.implementation = implementation
        
        if implementation == 'scratch':
            if model_type == 'ridge':
                self.model = RidgeRegressionFromScratch(**kwargs)
            elif model_type == 'lasso':
                self.model = LassoRegressionFromScratch(**kwargs)
            elif model_type == 'elasticnet':
                self.model = ElasticNetFromScratch(**kwargs)
            else:
                raise ValueError("Invalid model_type. Choose 'ridge', 'lasso', or 'elasticnet'.")
                
        else:  # sklearn
            if model_type == 'ridge':
                self.model = Ridge(**kwargs)
            elif model_type == 'lasso':
                self.model = Lasso(**kwargs)
            elif model_type == 'elasticnet':
                self.model = ElasticNet(**kwargs)
            else:
                raise ValueError("Invalid model_type. Choose 'ridge', 'lasso', or 'elasticnet'.")
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)
    
    def evaluate(self, X, y, true_coef=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : ndarray
            Test features
        y : ndarray
            True test targets
        true_coef : ndarray or None
            True coefficients if available
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Get coefficients based on implementation
        if self.implementation == 'scratch':
            coef = self.model.weights
            intercept = self.model.bias
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'coef': coef,
                'intercept': intercept,
                'cost_history': self.model.cost_history
            }
        else:  # sklearn
            coef = self.model.coef_
            intercept = self.model.intercept_
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'coef': coef,
                'intercept': intercept
            }
        
        # Calculate coefficient recovery metrics if true coefficients are provided
        if true_coef is not None:
            # Calculate percentage of correctly zeroed coefficients
            true_zero_mask = (true_coef == 0)
            pred_zero_mask = (coef == 0)
            
            # If there are any true zeros, calculate recovery rate
            if np.any(true_zero_mask):
                zero_recovery = np.mean(pred_zero_mask[true_zero_mask])
                metrics['zero_recovery'] = zero_recovery
            
            # If there are any non-zeros, calculate recovery rate
            if np.any(~true_zero_mask):
                nonzero_recovery = np.mean(~pred_zero_mask[~true_zero_mask])
                metrics['nonzero_recovery'] = nonzero_recovery
            
            # Calculate coefficient error
            metrics['coef_error'] = np.mean((coef - true_coef) ** 2)
        
        return metrics


def compare_regularization_paths(X, y, alphas=None):
    """
    Compare regularization paths for Ridge, Lasso, and ElasticNet.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    alphas : list or None
        List of regularization strengths to try
        
    Returns:
    --------
    paths : dict
        Dictionary containing coefficient paths for each model
    """
    if alphas is None:
        alphas = np.logspace(-5, 5, 100)
    
    # Initialize models
    ridge = Ridge()
    lasso = Lasso()
    elasticnet = ElasticNet(l1_ratio=0.5)
    
    # Compute regularization paths
    ridge_coefs = []
    lasso_coefs = []
    elasticnet_coefs = []
    
    for alpha in alphas:
        # Ridge
        ridge.set_params(alpha=alpha)
        ridge.fit(X, y)
        ridge_coefs.append(ridge.coef_.copy())
        
        # Lasso
        lasso.set_params(alpha=alpha)
        lasso.fit(X, y)
        lasso_coefs.append(lasso.coef_.copy())
        
        # ElasticNet
        elasticnet.set_params(alpha=alpha)
        elasticnet.fit(X, y)
        elasticnet_coefs.append(elasticnet.coef_.copy())
    
    # Convert to arrays
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    elasticnet_coefs = np.array(elasticnet_coefs)
    
    # Plot regularization paths
    plt.figure(figsize=(18, 5))
    
    # Ridge
    plt.subplot(1, 3, 1)
    plt.semilogx(alphas, ridge_coefs)
    plt.xlabel('alpha')
    plt.ylabel('Coefficients')
    plt.title('Ridge Regularization Path')
    plt.grid(True, alpha=0.3)
    
    # Lasso
    plt.subplot(1, 3, 2)
    plt.semilogx(alphas, lasso_coefs)
    plt.xlabel('alpha')
    plt.ylabel('Coefficients')
    plt.title('Lasso Regularization Path')
    plt.grid(True, alpha=0.3)
    
    # ElasticNet
    plt.subplot(1, 3, 3)
    plt.semilogx(alphas, elasticnet_coefs)
    plt.xlabel('alpha')
    plt.ylabel('Coefficients')
    plt.title('ElasticNet Regularization Path')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return paths
    paths = {
        'alphas': alphas,
        'ridge': ridge_coefs,
        'lasso': lasso_coefs,
        'elasticnet': elasticnet_coefs
    }
    
    return paths


def find_optimal_alpha(X, y, model_type='ridge', alphas=None, cv=5):
    """
    Find the optimal alpha value using cross-validation.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    model_type : str
        Type of regularization ('ridge', 'lasso', or 'elasticnet')
    alphas : list or None
        List of regularization strengths to try
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    best_alpha : float
        Best alpha value
    cv_results : dict
        Cross-validation results
    """
    if alphas is None:
        alphas = np.logspace(-5, 5, 100)
    
    # Initialize model based on type
    if model_type == 'ridge':
        model = Ridge()
        param_grid = {'alpha': alphas}
    elif model_type == 'lasso':
        model = Lasso()
        param_grid = {'alpha': alphas}
    elif model_type == 'elasticnet':
        model = ElasticNet()
        # For ElasticNet, we add l1_ratio as a hyperparameter
        param_grid = {'alpha': alphas, 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
    else:
        raise ValueError("Invalid model_type. Choose 'ridge', 'lasso', or 'elasticnet'.")
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='neg_mean_squared_error',
        return_train_score=True
    )
    grid_search.fit(X, y)
    
    # Get best parameters and CV results
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    
    # Extract scores for plotting
    if model_type != 'elasticnet':
        # Simple plot for Ridge or Lasso
        train_scores = -cv_results['mean_train_score']
        test_scores = -cv_results['mean_test_score']
        
        plt.semilogx(alphas, train_scores, 'b--', label='Training MSE')
        plt.semilogx(alphas, test_scores, 'g-', label='CV MSE')
        plt.axvline(x=best_params['alpha'], color='r', linestyle='--', 
                  label=f'Best alpha = {best_params["alpha"]:.4f}')
        
    else:
        # For ElasticNet, we plot the best alpha for each l1_ratio
        l1_ratios = np.array(param_grid['l1_ratio'])
        for l1_ratio in l1_ratios:
            mask = cv_results['param_l1_ratio'].data == l1_ratio
            train_scores = -cv_results['mean_train_score'][mask]
            test_scores = -cv_results['mean_test_score'][mask]
            
            plt.semilogx(alphas, test_scores, label=f'l1_ratio={l1_ratio}')
        
        plt.axvline(x=best_params['alpha'], color='r', linestyle='--', 
                  label=f'Best alpha = {best_params["alpha"]:.4f}\nl1_ratio = {best_params["l1_ratio"]}')
    
    plt.xlabel('alpha')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Cross-validation for {model_type.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Best parameters: {best_params}")
    if model_type == 'elasticnet':
        return best_params, cv_results
    else:
        return best_params['alpha'], cv_results


if __name__ == "__main__":
    # Example usage
    from data import generate_synthetic_data, preprocess_data
    
    # Generate synthetic data with multicollinearity
    print("Generating synthetic data with multicollinearity:")
    X, y, true_coef = generate_synthetic_data(n_samples=100, n_features=20, multicollinearity=True)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # Compare regularization paths
    print("\nComparing regularization paths:")
    paths = compare_regularization_paths(X_train, y_train)
    
    # Find optimal alpha for each model
    print("\nFinding optimal alpha for Ridge:")
    ridge_alpha, _ = find_optimal_alpha(X_train, y_train, model_type='ridge')
    
    print("\nFinding optimal alpha for Lasso:")
    lasso_alpha, _ = find_optimal_alpha(X_train, y_train, model_type='lasso')
    
    print("\nFinding optimal parameters for ElasticNet:")
    elasticnet_params, _ = find_optimal_alpha(X_train, y_train, model_type='elasticnet')
    
    # Train and evaluate models with optimal parameters
    print("\nTraining and evaluating models with optimal parameters:")
    
    # Ridge
    ridge_model = RegularizedRegressionModel('ridge', alpha=ridge_alpha)
    ridge_model.fit(X_train, y_train)
    ridge_metrics = ridge_model.evaluate(X_test, y_test, true_coef)
    print(f"Ridge MSE: {ridge_metrics['mse']:.4f}, R²: {ridge_metrics['r2']:.4f}")
    
    # Lasso
    lasso_model = RegularizedRegressionModel('lasso', alpha=lasso_alpha)
    lasso_model.fit(X_train, y_train)
    lasso_metrics = lasso_model.evaluate(X_test, y_test, true_coef)
    print(f"Lasso MSE: {lasso_metrics['mse']:.4f}, R²: {lasso_metrics['r2']:.4f}")
    
    # ElasticNet
    elasticnet_model = RegularizedRegressionModel(
        'elasticnet', 
        alpha=elasticnet_params['alpha'],
        l1_ratio=elasticnet_params['l1_ratio']
    )
    elasticnet_model.fit(X_train, y_train)
    elasticnet_metrics = elasticnet_model.evaluate(X_test, y_test, true_coef)
    print(f"ElasticNet MSE: {elasticnet_metrics['mse']:.4f}, R²: {elasticnet_metrics['r2']:.4f}")
    
    # Compare coefficient recovery
    print("\nCoefficient recovery comparison:")
    print(f"True non-zero coefficients: {np.sum(true_coef != 0)}/{len(true_coef)}")
    print(f"Ridge non-zero coefficients: {np.sum(ridge_metrics['coef'] != 0)}/{len(ridge_metrics['coef'])}")
    print(f"Lasso non-zero coefficients: {np.sum(lasso_metrics['coef'] != 0)}/{len(lasso_metrics['coef'])}")
    print(f"ElasticNet non-zero coefficients: {np.sum(elasticnet_metrics['coef'] != 0)}/{len(elasticnet_metrics['coef'])}")
    
    if 'zero_recovery' in lasso_metrics:
        print(f"Lasso zero recovery: {lasso_metrics['zero_recovery']:.2%}")
    if 'nonzero_recovery' in lasso_metrics:
        print(f"Lasso non-zero recovery: {lasso_metrics['nonzero_recovery']:.2%}")
