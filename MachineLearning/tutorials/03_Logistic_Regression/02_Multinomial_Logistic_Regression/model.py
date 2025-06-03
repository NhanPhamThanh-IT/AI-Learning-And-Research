#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multinomial logistic regression model implementation from scratch and using scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import OneHotEncoder


class MultinomialLogisticRegressionFromScratch:
    """
    Multinomial Logistic Regression implementation from scratch using gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, alpha=0.1, random_state=42):
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
        random_state : int
            Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes = None
        self.n_classes = None
        self.cost_history = []
    
    def _softmax(self, z):
        """
        Softmax function for multinomial classification.
        Computes the softmax of each row of the input z.
        
        Parameters:
        -----------
        z : ndarray of shape (n_samples, n_classes)
            Linear model output
            
        Returns:
        --------
        softmax_output : ndarray of shape (n_samples, n_classes)
            Softmax probabilities
        """
        # Shift z for numerical stability (to avoid overflow)
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """
        Convert class labels to one-hot encoded vectors.
        
        Parameters:
        -----------
        y : ndarray of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        y_one_hot : ndarray of shape (n_samples, n_classes)
            One-hot encoded target matrix
        """
        one_hot = np.zeros((y.shape[0], self.n_classes))
        for i, cls in enumerate(y):
            one_hot[i, cls] = 1
        return one_hot
    
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
        self : MultinomialLogisticRegressionFromScratch
            Fitted model
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Determine unique classes
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # One-hot encode target
        y_one_hot = self._one_hot_encode(y)
        
        # Initialize weights and bias
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)
        self.cost_history = []
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Linear model output
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Apply softmax function to get probabilities
            y_pred_proba = self._softmax(linear_model)
            
            # Compute gradients
            error = y_pred_proba - y_one_hot
            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error, axis=0)
            
            # Add regularization to gradients if specified
            if self.regularization == 'l2':  # Ridge
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == 'l1':  # Lasso
                dw += (self.alpha / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for history
            cost = self._compute_cost(y_one_hot, y_pred_proba)
            self.cost_history.append(cost)
            
            # Print progress
            if (iteration + 1) % 100 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def _compute_cost(self, y_one_hot, y_pred_proba):
        """
        Compute the cross-entropy cost.
        
        Parameters:
        -----------
        y_one_hot : ndarray of shape (n_samples, n_classes)
            One-hot encoded target matrix
        y_pred_proba : ndarray of shape (n_samples, n_classes)
            Predicted probabilities
            
        Returns:
        --------
        cost : float
            Cross-entropy cost
        """
        n_samples = y_one_hot.shape[0]
        
        # Compute cross-entropy loss
        log_likelihood = -np.sum(y_one_hot * np.log(y_pred_proba + 1e-15)) / n_samples
        
        # Add regularization term if specified
        if self.regularization == 'l2':  # Ridge
            reg_term = (self.alpha / (2 * n_samples)) * np.sum(np.square(self.weights))
            log_likelihood += reg_term
        elif self.regularization == 'l1':  # Lasso
            reg_term = (self.alpha / n_samples) * np.sum(np.abs(self.weights))
            log_likelihood += reg_term
            
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
        y : ndarray of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        score : float
            Mean accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def plot_cost_history(self):
        """
        Plot the cost history during training.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.n_iterations), self.cost_history)
        plt.title('Cost History During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True, alpha=0.3)
        plt.show()
        

class MultinomialLogisticRegressionSklearn:
    """
    Multinomial Logistic Regression implementation using scikit-learn.
    """
    
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42):
        """
        Initialize the model.
        
        Parameters:
        -----------
        penalty : {'l1', 'l2', 'elasticnet', None}
            Regularization type
        C : float
            Inverse of regularization strength (smaller values = stronger regularization)
        solver : str
            Algorithm to use in the optimization
        multi_class : {'auto', 'ovr', 'multinomial'}
            Strategy for multi-class classification
        max_iter : int
            Maximum number of iterations
        random_state : int
            Random seed for reproducibility
        """
        self.model = SklearnLogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            multi_class=multi_class,
            max_iter=max_iter,
            random_state=random_state
        )
        self.classes = None
    
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
        y : ndarray of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        self : MultinomialLogisticRegressionSklearn
            Fitted model
        """
        self.model.fit(X, y)
        self.classes = self.model.classes_
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
        y : ndarray of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        score : float
            Mean accuracy
        """
        return self.model.score(X, y)
    
    def get_coef(self):
        """
        Get the coefficients of the model.
        
        Returns:
        --------
        coef : ndarray of shape (n_features, n_classes)
            Coefficient of the features in the decision function
        intercept : ndarray of shape (n_classes,)
            Intercept (bias) term
        """
        return self.model.coef_, self.model.intercept_


def evaluate_classification_model(model, X_test, y_test, target_names=None):
    """
    Evaluate a classification model with various metrics.
    
    Parameters:
    -----------
    model : object
        Trained classification model with predict method
    X_test : ndarray
        Test feature matrix
    y_test : ndarray
        Test target vector
    target_names : list or None
        List of class names
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Class labels
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    if target_names is None:
        target_names = [f"Class {i}" for i in range(n_classes)]
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_micro': recall_score(y_test, y_pred, average='micro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_micro': f1_score(y_test, y_pred, average='micro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'conf_matrix': confusion_matrix(y_test, y_pred),
    }
    
    # Print evaluation results
    print("\nModel Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nPrecision (micro): {metrics['precision_micro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"\nRecall (micro): {metrics['recall_micro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"\nF1 Score (micro): {metrics['f1_micro']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return metrics


def plot_decision_regions(X, y, model, feature_names=None, class_names=None, resolution=0.02):
    """
    Plot decision regions for the first two features of the dataset.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target vector
    model : object
        Trained model with predict method
    feature_names : list or None
        List of feature names
    class_names : list or None
        List of class names
    resolution : float
        Grid resolution for plotting decision boundaries
    """
    # Check if we have more than 2 features
    if X.shape[1] > 2:
        print("Warning: Using only the first two features for visualization")
        X_vis = X[:, :2]
    else:
        X_vis = X
    
    # Set feature names
    if feature_names is None or len(feature_names) < 2:
        feature_names = [f"Feature {i+1}" for i in range(2)]
    
    # Set class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y)))]
    
    # Plot setup
    plt.figure(figsize=(12, 10))
    
    # Define the grid
    x1_min, x1_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    x2_min, x2_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Create the test points for the grid
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    
    # If original data has more features, pad with zeros
    if X.shape[1] > 2:
        grid_pad = np.zeros((grid_points.shape[0], X.shape[1] - 2))
        grid_points = np.column_stack([grid_points, grid_pad])
    
    # Get predictions on the grid
    Z = model.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # Plot decision boundaries
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class samples
    for i, label in enumerate(np.unique(y)):
        plt.scatter(
            X_vis[y == label, 0], 
            X_vis[y == label, 1],
            alpha=0.8, 
            label=class_names[i],
            edgecolor='k'
        )
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Import data functions
    from data import generate_synthetic_data, preprocess_data, visualize_data_2d
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=300, n_classes=3, n_features=2, class_sep=1.5)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2)
    
    # Visualize data
    visualize_data_2d(X, y, title="Synthetic Multi-class Classification Data")
    
    print("\n1. Training Multinomial Logistic Regression from Scratch:")
    # Train model from scratch
    model_scratch = MultinomialLogisticRegressionFromScratch(
        learning_rate=0.1, 
        n_iterations=1000,
        regularization='l2',
        alpha=0.01
    )
    model_scratch.fit(X_train, y_train)
    model_scratch.plot_cost_history()
    
    # Evaluate model
    print("\nEvaluating scratch model:")
    print(f"Training accuracy: {model_scratch.score(X_train, y_train):.4f}")
    print(f"Testing accuracy: {model_scratch.score(X_test, y_test):.4f}")
    
    print("\n2. Training Scikit-learn Multinomial Logistic Regression:")
    # Train scikit-learn model
    model_sklearn = MultinomialLogisticRegressionSklearn(
        penalty='l2',
        C=1.0,
        multi_class='multinomial'
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating sklearn model:")
    print(f"Training accuracy: {model_sklearn.score(X_train, y_train):.4f}")
    print(f"Testing accuracy: {model_sklearn.score(X_test, y_test):.4f}")
    
    # Full evaluation
    evaluate_classification_model(model_sklearn, X_test, y_test)
    
    # Plot decision regions
    plot_decision_regions(X, y, model_sklearn)
