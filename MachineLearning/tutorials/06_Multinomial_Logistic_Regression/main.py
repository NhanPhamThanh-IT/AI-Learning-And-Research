#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for Multinomial Logistic Regression tutorial.

This script demonstrates the complete analysis pipeline for multinomial logistic regression
using both custom implementation and scikit-learn.
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
    visualize_decision_boundaries,
    visualize_feature_importance,
    create_polynomial_features,
    load_data
)

from model import (
    MultinomialLogisticRegressionFromScratch,
    MultinomialLogisticRegressionSklearn,
    evaluate_classification_model,
    plot_decision_regions,
    MultinomialLogisticRegression
)


def run_synthetic_data_example(n_samples=300, n_features=2, n_classes=3, test_size=0.2,
                             learning_rate=0.1, n_iterations=1000, regularization='l2',
                             alpha=0.01, class_sep=1.5, visualize=True, random_state=42):
    """
    Run a complete pipeline with synthetic data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    test_size : float
        Proportion of the dataset to include in the test split
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of iterations for training
    regularization : str or None
        Type of regularization ('l1', 'l2', or None)
    alpha : float
        Regularization strength
    class_sep : float
        Class separation factor
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("MULTINOMIAL LOGISTIC REGRESSION WITH SYNTHETIC DATA")
    print("="*80)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=n_classes,
        class_sep=class_sep, 
        random_state=random_state
    )
    
    # Visualize data
    if visualize and n_features <= 5:
        visualize_data_2d(X, y, title="Synthetic Multi-class Classification Data")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in testing: {np.bincount(y_test)}")
    
    # Train custom model
    print("\nTraining model from scratch...")
    from_scratch_model = MultinomialLogisticRegressionFromScratch(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization,
        alpha=alpha,
        random_state=random_state
    )
    from_scratch_model.fit(X_train, y_train)
    
    # Evaluate custom model
    print("\nEvaluating model from scratch:")
    train_accuracy = from_scratch_model.score(X_train, y_train)
    test_accuracy = from_scratch_model.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    
    if visualize:
        from_scratch_model.plot_cost_history()
        if n_features == 2:
            plot_decision_regions(X, y, from_scratch_model)
    
    # Train scikit-learn model
    print("\nTraining scikit-learn model...")
    sklearn_model = MultinomialLogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else 1.0,
        multi_class='multinomial',
        max_iter=n_iterations,
        random_state=random_state
    )
    sklearn_model.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    sklearn_train_accuracy = sklearn_model.score(X_train, y_train)
    sklearn_test_accuracy = sklearn_model.score(X_test, y_test)
    print(f"Training accuracy: {sklearn_train_accuracy:.4f}")
    print(f"Testing accuracy: {sklearn_test_accuracy:.4f}")
    
    # Full evaluation with metrics
    metrics = evaluate_classification_model(sklearn_model, X_test, y_test)
    
    if visualize and n_features == 2:
        plot_decision_regions(X, y, sklearn_model)
    
    # Display model coefficients
    if visualize:
        coef, intercept = sklearn_model.get_coef()
        visualize_feature_importance(coef, 
                                    feature_names=[f"Feature {i+1}" for i in range(n_features)],
                                    title="Feature Importance by Class")
    
    return from_scratch_model, sklearn_model, metrics


def run_real_data_example(dataset_name='iris', test_size=0.2, learning_rate=0.1, n_iterations=1000,
                        regularization='l2', alpha=0.01, visualize=True, random_state=42):
    """
    Run a complete pipeline with a real dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('iris' or 'digits')
    test_size : float
        Proportion of the dataset to include in the test split
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of iterations for training
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
    print(f"MULTINOMIAL LOGISTIC REGRESSION WITH {dataset_name.upper()} DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names, target_names = load_real_data(dataset_name)
    
    # Visualize data
    if visualize:
        if X.shape[1] <= 5:  # For low-dimensional data
            visualize_data_2d(X, y, title=f"{dataset_name.title()} Dataset", target_names=target_names)
        else:
            # For high-dimensional data, use only first few features for visualization
            visualize_data_2d(X[:, :5], y, title=f"{dataset_name.title()} Dataset (First 5 Features)", 
                             target_names=target_names)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train scikit-learn model
    print("\nTraining scikit-learn model...")
    sklearn_model = MultinomialLogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else 1.0,
        multi_class='multinomial',
        max_iter=n_iterations,
        random_state=random_state
    )
    sklearn_model.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    sklearn_train_accuracy = sklearn_model.score(X_train, y_train)
    sklearn_test_accuracy = sklearn_model.score(X_test, y_test)
    print(f"Training accuracy: {sklearn_train_accuracy:.4f}")
    print(f"Testing accuracy: {sklearn_test_accuracy:.4f}")
    
    # Full evaluation with metrics
    metrics = evaluate_classification_model(sklearn_model, X_test, y_test, target_names=target_names)
    
    if visualize and X.shape[1] == 2:
        plot_decision_regions(X, y, sklearn_model, feature_names=feature_names, class_names=target_names)
    
    # Display model coefficients
    if visualize:
        coef, intercept = sklearn_model.get_coef()
        visualize_feature_importance(coef, feature_names=feature_names,
                                   title=f"Feature Importance by Class - {dataset_name.title()} Dataset")
    
    return sklearn_model, metrics


def run_custom_file_example(filename, target_col=-1, test_size=0.2, learning_rate=0.1, n_iterations=1000,
                         regularization='l2', alpha=0.01, visualize=True, random_state=42):
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
        Learning rate for gradient descent
    n_iterations : int
        Number of iterations for training
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
    print(f"MULTINOMIAL LOGISTIC REGRESSION WITH CUSTOM DATA FROM {filename}")
    print("="*80)
    
    # Load custom data
    print(f"\nLoading data from {filename}...")
    X, y, feature_names, target_names = load_custom_data(filename, target_col=target_col)
    
    # Rest of the pipeline is similar to the real data example
    if visualize:
        if X.shape[1] <= 5:  # For low-dimensional data
            visualize_data_2d(X, y, title=f"Custom Dataset from {filename}", target_names=target_names)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train scikit-learn model
    print("\nTraining scikit-learn model...")
    sklearn_model = MultinomialLogisticRegressionSklearn(
        penalty=regularization,
        C=1.0/alpha if alpha > 0 else 1.0,
        multi_class='multinomial',
        max_iter=n_iterations,
        random_state=random_state
    )
    sklearn_model.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics = evaluate_classification_model(sklearn_model, X_test, y_test, target_names=target_names)
    
    if visualize and X.shape[1] == 2:
        plot_decision_regions(X, y, sklearn_model, feature_names=feature_names, class_names=target_names)
    
    # Display model coefficients
    if visualize:
        coef, intercept = sklearn_model.get_coef()
        visualize_feature_importance(coef, feature_names=feature_names,
                                   title=f"Feature Importance by Class")
    
    return sklearn_model, metrics


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Multinomial Logistic Regression Tutorial')
    
    parser.add_argument('--data-type', type=str, default='synthetic',
                      choices=['synthetic', 'iris', 'digits', 'custom'],
                      help='Type of data to use (synthetic, iris, digits, custom)')
    
    parser.add_argument('--file', type=str, default=None,
                      help='Path to custom data file (when data-type is custom)')
    
    parser.add_argument('--target-col', type=int, default=-1,
                      help='Column index of target variable (for custom data)')
    
    parser.add_argument('--n-samples', type=int, default=300,
                      help='Number of samples for synthetic data')
    
    parser.add_argument('--n-features', type=int, default=2,
                      help='Number of features for synthetic data')
    
    parser.add_argument('--n-classes', type=int, default=3,
                      help='Number of classes for synthetic data')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--learning-rate', type=float, default=0.1,
                      help='Learning rate for gradient descent')
    
    parser.add_argument('--n-iterations', type=int, default=1000,
                      help='Number of iterations for training')
    
    parser.add_argument('--regularization', type=str, default='l2',
                      choices=['l1', 'l2', 'none'],
                      help='Type of regularization')
    
    parser.add_argument('--alpha', type=float, default=0.01,
                      help='Regularization strength')
    
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                      help='Disable visualizations')
    
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Convert 'none' string to None for regularization
    if args.regularization == 'none':
        args.regularization = None
    
    # Run the appropriate example based on data type
    if args.data_type == 'synthetic':
        run_synthetic_data_example(
            n_samples=args.n_samples, 
            n_features=args.n_features, 
            n_classes=args.n_classes,
            test_size=args.test_size,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.data_type in ['iris', 'digits']:
        run_real_data_example(
            dataset_name=args.data_type,
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
    
    print("\nMultinomial Logistic Regression Tutorial completed successfully!")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = MultinomialLogisticRegression()
    model.fit(X_train, y_train)
    acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")
