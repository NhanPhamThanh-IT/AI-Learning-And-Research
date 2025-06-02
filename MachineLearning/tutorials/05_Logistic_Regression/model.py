#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logistic regression model implementation from scratch and using scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, roc_auc_score, confusion_matrix, classification_report
)


class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation from scratch using gradient descent.
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
    
    def _sigmoid(self, z):
        """Sigmoid (logistic) function."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
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
        self : LogisticRegressionFromScratch
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model output
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_pred = self._sigmoid(linear_model)
            
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
        """Compute the logistic cost function (binary cross-entropy with optional regularization)."""
        n_samples = X.shape[0]
        
        # Avoid log(0) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization if specified
        if self.regularization == 'l2':  # Ridge
            cost += (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':  # Lasso
            cost += (self.alpha / n_samples) * np.sum(np.abs(self.weights))
        
        return cost
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples,)
            Probability of class 1
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_proba = self._sigmoid(linear_model)
        return y_proba
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples
        threshold : float
            Classification threshold
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        return y_pred


class LogisticRegressionModel:
    """
    A wrapper class providing both from-scratch and scikit-learn implementations
    of logistic regression.
    """
    
    def __init__(self, implementation='sklearn', **kwargs):
        """
        Initialize the logistic regression model.
        
        Parameters:
        -----------
        implementation : str
            Which implementation to use ('scratch' or 'sklearn')
        **kwargs : dict
            Additional parameters for the specific implementation
        """
        self.implementation = implementation
        
        if implementation == 'scratch':
            self.model = LogisticRegressionFromScratch(**kwargs)
        else:  # sklearn
            # Convert regularization to sklearn format
            if 'regularization' in kwargs:
                if kwargs['regularization'] == 'l1':
                    kwargs['penalty'] = 'l1'
                    kwargs['solver'] = 'liblinear'  # liblinear supports l1
                elif kwargs['regularization'] == 'l2':
                    kwargs['penalty'] = 'l2'
                elif kwargs['regularization'] is None:
                    kwargs['penalty'] = None
                del kwargs['regularization']
            
            # Convert alpha to C (C = 1/alpha)
            if 'alpha' in kwargs:
                if kwargs['alpha'] > 0:
                    kwargs['C'] = 1 / kwargs['alpha']
                del kwargs['alpha']
            
            self.model = SklearnLogisticRegression(**kwargs)
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.implementation == 'scratch':
            return self.model.predict_proba(X)
        else:  # sklearn
            # sklearn returns probabilities for both classes
            return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        if self.implementation == 'scratch':
            return self.model.predict(X, threshold)
        else:  # sklearn
            if threshold == 0.5:
                return self.model.predict(X)
            else:
                proba = self.predict_proba(X)
                return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : ndarray
            Test features
        y : ndarray
            True test targets
        threshold : float
            Classification threshold
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        # Get predictions and probabilities
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Get coefficients based on implementation
        if self.implementation == 'scratch':
            coef = self.model.weights
            intercept = self.model.bias
            cost_history = self.model.cost_history
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'confusion_matrix': cm,
                'coef': coef,
                'intercept': intercept,
                'cost_history': cost_history
            }
        else:  # sklearn
            coef = self.model.coef_[0]  # For binary classification
            intercept = self.model.intercept_[0]
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'confusion_matrix': cm,
                'coef': coef,
                'intercept': intercept
            }
        
        return metrics


def find_optimal_threshold(y_true, y_proba):
    """
    Find the optimal classification threshold based on F1 score.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_proba : ndarray
        Predicted probabilities
        
    Returns:
    --------
    optimal_threshold : float
        Optimal threshold value
    thresholds_metrics : dict
        Metrics for various thresholds
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
    
    # Find threshold with highest F1 score
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Plot precision-recall vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b--', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-.', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
              label=f'Optimal threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    thresholds_metrics = {
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'precisions': precisions,
        'recalls': recalls,
        'optimal_threshold': optimal_threshold
    }
    
    return optimal_threshold, thresholds_metrics


def plot_roc_curve(fpr, tpr, auc, title="Receiver Operating Characteristic (ROC) Curve"):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    auc : float
        Area under the ROC curve
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : ndarray of shape (2, 2)
        Confusion matrix
    class_names : list or None
        List of class names
    title : str
        Plot title
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot learning curve to diagnose bias-variance tradeoff.
    
    Parameters:
    -----------
    model : object
        Model object with fit and predict methods
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of jobs to run in parallel
    train_sizes : ndarray
        Training set sizes to plot the learning curves
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation accuracy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from data import generate_synthetic_data, preprocess_data, visualize_data_2d, visualize_decision_boundary
    
    # Generate synthetic data for binary classification
    X, y = generate_synthetic_data(n_samples=200, n_features=2, class_sep=1.5)
    
    # Visualize data
    visualize_data_2d(X, y, "Synthetic Binary Classification Data")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # Train model
    print("Training logistic regression model...")
    model = LogisticRegressionModel('sklearn')
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    y_proba = model.predict_proba(X_test)
    optimal_threshold, _ = find_optimal_threshold(y_test, y_proba)
    
    # Re-evaluate with optimal threshold
    metrics_optimal = model.evaluate(X_test, y_test, threshold=optimal_threshold)
    print(f"\nModel evaluation with optimal threshold ({optimal_threshold:.2f}):")
    print(f"Accuracy: {metrics_optimal['accuracy']:.4f}")
    print(f"Precision: {metrics_optimal['precision']:.4f}")
    print(f"Recall: {metrics_optimal['recall']:.4f}")
    print(f"F1 Score: {metrics_optimal['f1']:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Visualize decision boundary
    visualize_decision_boundary(X, y, model, "Logistic Regression Decision Boundary")
