#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for Logistic Regression tutorial.

This script demonstrates the complete analysis pipeline for logistic regression,
including data generation, model training, evaluation, and visualization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import (
    generate_synthetic_data,
    load_real_data,
    load_custom_data,
    preprocess_data,
    visualize_data_2d,
    visualize_decision_boundary
)

from model import (
    LogisticRegressionFromScratch,
    LogisticRegressionSklearn,
    evaluate_classification_model,
    plot_roc_curve,
    plot_decision_regions
)


def run_synthetic_data_example(n_samples=200, n_features=2, test_size=0.2,
                             class_sep=1.0, noise=0.1, learning_rate=0.01, n_iterations=1000,
                             regularization=None, alpha=0.1, visualize=True, random_state=42):
    """
    Run a complete pipeline with synthetic data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Proportion of the dataset to include in the test split
    class_sep : float
        Class separation factor
    noise : float
        Noise level for synthetic data
    learning_rate : float
        Learning rate for gradient descent (for scratch implementation)
    n_iterations : int
        Number of iterations for training (for scratch implementation)
    regularization : str or None
        Type of regularization ('l1', 'l2', or None)
    alpha : float
        Regularization strength
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION WITH SYNTHETIC DATA")
    print("="*80)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data with {n_features} features...")
    X, y = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        class_sep=class_sep,
        noise=noise,
        random_state=random_state
    )
    
    # Visualize data
    if visualize and n_features <= 2:
        visualize_data_2d(X, y, title="Synthetic Binary Classification Data")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in testing: {np.bincount(y_test)}")
    
    # Train logistic regression model from scratch
    print("\nTraining logistic regression model from scratch...")
    model_scratch = LogisticRegressionFromScratch(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization,
        alpha=alpha
    )
    model_scratch.fit(X_train, y_train)
    
    # Evaluate scratch model
    print("\nEvaluating model from scratch:")
    train_accuracy = model_scratch.score(X_train, y_train)
    test_accuracy = model_scratch.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    
    if visualize:
        # Plot cost history
        model_scratch.plot_cost_history()
        
        if n_features == 2:
            # Plot decision boundary
            plot_decision_regions(X, y, model_scratch)
    
    # Train scikit-learn model
    print("\nTraining scikit-learn logistic regression model...")
    model_sklearn = LogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else float('inf'),
        solver='liblinear' if regularization == 'l1' else 'lbfgs',
        random_state=random_state
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics = evaluate_classification_model(model_sklearn, X_test, y_test)
    
    if visualize:
        if n_features == 2:
            # Plot decision boundary
            plot_decision_regions(X, y, model_sklearn, title="Scikit-learn Logistic Regression Decision Boundary")
        
        # Plot ROC curve
        plot_roc_curve(model_sklearn, X_test, y_test)
    
    return model_sklearn, metrics


def run_real_data_example(dataset_name='breast_cancer', test_size=0.2,
                        learning_rate=0.01, n_iterations=1000, regularization=None,
                        alpha=0.1, visualize=True, random_state=42):
    """
    Run a complete pipeline with a real dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('breast_cancer', 'heart', 'diabetes')
    test_size : float
        Proportion of the dataset to include in the test split
    learning_rate : float
        Learning rate for gradient descent (for scratch implementation)
    n_iterations : int
        Number of iterations for training (for scratch implementation)
    regularization : str or None
        Type of regularization ('l1', 'l2', or None)
    alpha : float
        Regularization strength
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print(f"LOGISTIC REGRESSION WITH {dataset_name.upper()} DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names, target_names = load_real_data(dataset_name)
    
    # Visualize data
    if visualize:
        if X.shape[1] <= 5:  # For low-dimensional data
            visualize_data_2d(X, y, title=f"{dataset_name.title()} Dataset")
        else:
            # For high-dimensional data, use PCA or feature selection
            from sklearn.decomposition import PCA
            print("Applying PCA for visualization...")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_.sum()
            visualize_data_2d(X_pca, y, 
                             title=f"{dataset_name.title()} Dataset (PCA, {explained_variance:.2%} variance explained)")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train scikit-learn model
    print("\nTraining scikit-learn logistic regression model...")
    model_sklearn = LogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else float('inf'),
        solver='liblinear' if regularization == 'l1' else 'lbfgs',
        random_state=random_state
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics = evaluate_classification_model(model_sklearn, X_test, y_test, target_names=target_names)
    
    if visualize:
        # Plot ROC curve
        plot_roc_curve(model_sklearn, X_test, y_test)
        
        # Plot feature importance
        coef = model_sklearn.coef_[0]
        idx = np.argsort(np.abs(coef))
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), coef[idx], align='center')
        plt.yticks(range(len(idx)), np.array(feature_names)[idx])
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # If using PCA, show the decision boundary in the reduced space
        if X.shape[1] > 2:
            # Create a new model for the PCA-transformed data
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            model_pca = LogisticRegressionSklearn(
                penalty=regularization,
                C=1.0/alpha if alpha > 0 else float('inf'),
                random_state=random_state
            )
            model_pca.fit(X_train_pca, y_train)
            pca_accuracy = model_pca.score(X_test_pca, y_test)
            print(f"\nPCA model accuracy: {pca_accuracy:.4f}")
            
            # Visualize decision boundary in 2D PCA space
            X_pca = np.vstack([X_train_pca, X_test_pca])
            y_full = np.hstack([y_train, y_test])
            plot_decision_regions(X_pca, y_full, model_pca, 
                                title=f"Decision Boundary in PCA Space (Accuracy: {pca_accuracy:.4f})")
    
    return model_sklearn, metrics


def run_custom_file_example(filename, target_col=-1, test_size=0.2, learning_rate=0.01, 
                         n_iterations=1000, regularization=None, alpha=0.1, 
                         visualize=True, random_state=42):
    """
    Run a complete pipeline with custom data from a file.
    
    Parameters:
    -----------
    filename : str
        Path to the data file
    target_col : int
        Column index for the target variable
    test_size : float
        Proportion of the dataset to include in the test split
    learning_rate : float
        Learning rate for gradient descent (for scratch implementation)
    n_iterations : int
        Number of iterations for training (for scratch implementation)
    regularization : str or None
        Type of regularization ('l1', 'l2', or None)
    alpha : float
        Regularization strength
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print(f"LOGISTIC REGRESSION WITH CUSTOM DATA FROM {filename}")
    print("="*80)
    
    # Load custom data
    print(f"\nLoading data from {filename}...")
    X, y, feature_names, target_names = load_custom_data(filename, target_col=target_col)
    
    # Rest of the pipeline is similar to the real data example
    if visualize:
        if X.shape[1] <= 2:  # For low-dimensional data
            visualize_data_2d(X, y, title=f"Custom Dataset from {filename}")
        else:
            # For high-dimensional data, use PCA
            from sklearn.decomposition import PCA
            print("Applying PCA for visualization...")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_.sum()
            visualize_data_2d(X_pca, y, 
                             title=f"Custom Dataset (PCA, {explained_variance:.2%} variance explained)")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train scikit-learn model
    print("\nTraining scikit-learn logistic regression model...")
    model_sklearn = LogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else float('inf'),
        solver='liblinear' if regularization == 'l1' else 'lbfgs',
        random_state=random_state
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics = evaluate_classification_model(model_sklearn, X_test, y_test, target_names=target_names)
    
    if visualize:
        # Plot ROC curve
        plot_roc_curve(model_sklearn, X_test, y_test)
        
        # If 2D data, visualize decision boundary
        if X.shape[1] == 2:
            plot_decision_regions(X, y, model_sklearn, 
                                title="Logistic Regression Decision Boundary")
    
    return model_sklearn, metrics


def run_regularization_comparison(n_samples=200, n_features=20, test_size=0.2,
                               noise=0.5, visualize=True, random_state=42):
    """
    Compare different regularization methods for logistic regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Proportion of the dataset to include in the test split
    noise : float
        Noise level for synthetic data
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("COMPARING REGULARIZATION METHODS FOR LOGISTIC REGRESSION")
    print("="*80)
    
    # Generate synthetic data with many features but only some informative
    print(f"\nGenerating synthetic data with {n_features} features...")
    X, y = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=int(n_features / 4),  # Only 25% of features are informative
        noise=noise,
        random_state=random_state
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Compare different regularization methods
    models = {}
    metrics = {}
    
    # No regularization
    print("\n1. Logistic Regression (No Regularization)")
    model_none = LogisticRegressionSklearn(
        penalty=None,
        solver='lbfgs',
        random_state=random_state
    )
    model_none.fit(X_train, y_train)
    metrics['None'] = evaluate_classification_model(model_none, X_test, y_test, 
                                               model_name="Logistic Regression (No Regularization)")
    models['None'] = model_none
    
    # L2 regularization (Ridge)
    print("\n2. Logistic Regression with L2 Regularization (Ridge)")
    model_l2 = LogisticRegressionSklearn(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        random_state=random_state
    )
    model_l2.fit(X_train, y_train)
    metrics['L2'] = evaluate_classification_model(model_l2, X_test, y_test, 
                                             model_name="Logistic Regression (L2 Regularization)")
    models['L2'] = model_l2
    
    # L1 regularization (Lasso)
    print("\n3. Logistic Regression with L1 Regularization (Lasso)")
    model_l1 = LogisticRegressionSklearn(
        penalty='l1',
        C=1.0,
        solver='liblinear',
        random_state=random_state
    )
    model_l1.fit(X_train, y_train)
    metrics['L1'] = evaluate_classification_model(model_l1, X_test, y_test, 
                                             model_name="Logistic Regression (L1 Regularization)")
    models['L1'] = model_l1
    
    # Elasticnet regularization (L1 + L2)
    print("\n4. Logistic Regression with Elasticnet Regularization (L1 + L2)")
    model_elasticnet = LogisticRegressionSklearn(
        penalty='elasticnet',
        C=1.0,
        solver='saga',
        l1_ratio=0.5,
        random_state=random_state
    )
    model_elasticnet.fit(X_train, y_train)
    metrics['Elasticnet'] = evaluate_classification_model(model_elasticnet, X_test, y_test, 
                                                     model_name="Logistic Regression (Elasticnet Regularization)")
    models['Elasticnet'] = model_elasticnet
    
    if visualize:
        # Compare coefficients
        plt.figure(figsize=(12, 8))
        plt.plot(models['None'].coef_[0], 'o', label='No regularization')
        plt.plot(models['L2'].coef_[0], '^', label='L2 (Ridge)')
        plt.plot(models['L1'].coef_[0], 's', label='L1 (Lasso)')
        plt.plot(models['Elasticnet'].coef_[0], 'd', label='Elasticnet')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('Coefficient Values with Different Regularization Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Count non-zero coefficients
        non_zero_counts = {
            'No regularization': np.sum(np.abs(models['None'].coef_[0]) > 1e-10),
            'L2 (Ridge)': np.sum(np.abs(models['L2'].coef_[0]) > 1e-10),
            'L1 (Lasso)': np.sum(np.abs(models['L1'].coef_[0]) > 1e-10),
            'Elasticnet': np.sum(np.abs(models['Elasticnet'].coef_[0]) > 1e-10)
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(non_zero_counts.keys(), non_zero_counts.values())
        plt.ylabel('Number of Non-Zero Coefficients')
        plt.title('Feature Selection Effect of Different Regularization Methods')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        # Compare model performance metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        performance = {}
        
        for metric in metric_names:
            performance[metric] = [metrics[m][metric] for m in metrics]
            
        plt.figure(figsize=(15, 12))
        for i, metric in enumerate(metric_names):
            plt.subplot(len(metric_names), 1, i + 1)
            plt.bar(metrics.keys(), performance[metric])
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    return models, metrics


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Logistic Regression Tutorial')
    
    parser.add_argument('--data-type', type=str, default='synthetic',
                      choices=['synthetic', 'real', 'custom', 'regularization'],
                      help='Type of data to use (synthetic, real, custom, regularization)')
    
    parser.add_argument('--dataset', type=str, default='breast_cancer',
                      choices=['breast_cancer', 'heart', 'diabetes'],
                      help='Dataset name for real data')
    
    parser.add_argument('--file', type=str, default=None,
                      help='Path to custom data file (when data-type is custom)')
    
    parser.add_argument('--target-col', type=int, default=-1,
                      help='Column index of target variable (for custom data)')
    
    parser.add_argument('--n-samples', type=int, default=200,
                      help='Number of samples for synthetic data')
    
    parser.add_argument('--n-features', type=int, default=2,
                      help='Number of features for synthetic data')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--class-sep', type=float, default=1.0,
                      help='Class separation factor for synthetic data')
    
    parser.add_argument('--noise', type=float, default=0.1,
                      help='Noise level for synthetic data')
    
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate for gradient descent')
    
    parser.add_argument('--n-iterations', type=int, default=1000,
                      help='Number of iterations for training')
    
    parser.add_argument('--regularization', type=str, default=None,
                      choices=['l1', 'l2', None],
                      help='Type of regularization')
    
    parser.add_argument('--alpha', type=float, default=0.1,
                      help='Regularization strength')
    
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                      help='Disable visualizations')
    
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Run the appropriate example based on data type
    if args.data_type == 'synthetic':
        run_synthetic_data_example(
            n_samples=args.n_samples, 
            n_features=args.n_features, 
            test_size=args.test_size,
            class_sep=args.class_sep,
            noise=args.noise,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.data_type == 'real':
        run_real_data_example(
            dataset_name=args.dataset,
            test_size=args.test_size,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.data_type == 'custom':
        if args.file is None:
            print("Error: --file argument is required for custom data type")
            return
            
        run_custom_file_example(
            filename=args.file,
            target_col=args.target_col,
            test_size=args.test_size,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.data_type == 'regularization':
        run_regularization_comparison(
            n_samples=args.n_samples,
            n_features=20,  # Use more features for regularization comparison
            test_size=args.test_size,
            noise=args.noise,
            visualize=args.visualize,
            random_state=args.random_state
        )
    
    print("\nLogistic Regression Tutorial completed successfully!")


if __name__ == "__main__":
    main()
