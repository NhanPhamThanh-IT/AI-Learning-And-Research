#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running multiple linear regression analysis and examples.
"""

import argparse
import matplotlib.pyplot as plt
from data import generate_synthetic_data, load_real_data, load_custom_data, preprocess_data, analyze_features
from model import MultipleRegressionModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multiple Linear Regression Analysis')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'california', 'custom'],
                       help='Dataset to use: synthetic, california, or custom')
    
    parser.add_argument('--custom_file', type=str, default=None,
                       help='Path to custom CSV file if dataset=custom')
    
    parser.add_argument('--n_samples', type=int, default=200,
                       help='Number of samples for synthetic data')
    
    parser.add_argument('--n_features', type=int, default=3,
                       help='Number of features for synthetic data')
    
    parser.add_argument('--noise', type=float, default=0.5,
                       help='Noise level for synthetic data')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--standardize', action='store_true',
                       help='Whether to standardize features')
    
    parser.add_argument('--implementation', type=str, default='sklearn',
                       choices=['sklearn', 'scratch'],
                       help='Implementation to use: sklearn or scratch')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for gradient descent (scratch implementation)')
    
    parser.add_argument('--n_iterations', type=int, default=2000,
                       help='Number of iterations for gradient descent (scratch implementation)')
    
    parser.add_argument('--regularization', type=str, default=None,
                       choices=[None, 'l1', 'l2'],
                       help='Type of regularization: None, l1, or l2')
    
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Regularization strength')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Whether to perform feature analysis')
    
    parser.add_argument('--plot_residuals', action='store_true',
                       help='Whether to plot residual analysis')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Whether to print detailed information')
    
    return parser.parse_args()


def run_analysis(args):
    """
    Run the multiple regression analysis with specified arguments.
    
    Parameters:
    -----------
    args : Namespace
        Command line arguments
    """
    # Load or generate the data
    if args.dataset == 'synthetic':
        X, y, true_coef = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise
        )
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        if args.verbose:
            print(f"Generated synthetic data with {X.shape[0]} samples and {X.shape[1]} features")
            print(f"True coefficients: {true_coef}")
    
    elif args.dataset == 'california':
        X, y, feature_names = load_real_data()
        if args.verbose:
            print(f"Loaded California housing dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    elif args.dataset == 'custom':
        if args.custom_file is None:
            print("Error: --custom_file must be specified when using custom dataset")
            return
        X, y, feature_names = load_custom_data(args.custom_file)
        if args.verbose:
            print(f"Loaded custom dataset from {args.custom_file} with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Perform feature analysis if requested
    if args.analyze:
        analyze_features(X, y, feature_names)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, 
        test_size=args.test_size, 
        standardize=args.standardize
    )
    
    if args.verbose:
        print(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        if args.standardize:
            print("Features standardized")
    
    # Create and train the model
    model = MultipleRegressionModel(
        implementation=args.implementation,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        regularization=args.regularization,
        alpha=args.alpha
    )
    
    if args.verbose:
        print(f"Training {args.implementation} implementation...")
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test, feature_names)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    print(f"Adjusted R-squared: {metrics['adjusted_r2']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    
    if args.implementation == 'scratch':
        print("\nModel Parameters:")
        print(f"Intercept (bias): {metrics['bias']:.4f}")
        print("Coefficients:")
        for i, (name, weight) in enumerate(zip(feature_names, metrics['weights'])):
            print(f"  {name}: {weight:.4f}")
        
        # Plot cost history
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['cost_history'])
        plt.title('Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
        
    else:  # sklearn
        print("\nModel Parameters:")
        print(f"Intercept: {metrics['intercept']:.4f}")
        print("Coefficients:")
        for i, (name, coef) in enumerate(zip(feature_names, metrics['coef'])):
            print(f"  {name}: {coef:.4f}")
    
    # Plot residuals if requested
    if args.plot_residuals:
        model.plot_residuals(X_test, y_test)
    
    # Calculate and print predictions for a few samples
    if args.verbose:
        n_samples_to_show = min(5, X_test.shape[0])
        print("\nSample predictions:")
        y_pred = model.predict(X_test[:n_samples_to_show])
        
        for i in range(n_samples_to_show):
            print(f"Sample {i+1}:")
            print(f"  Features: {X_test[i]}")
            print(f"  Actual: {y_test[i]:.4f}, Predicted: {y_pred[i]:.4f}, Error: {y_test[i] - y_pred[i]:.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    run_analysis(args)
