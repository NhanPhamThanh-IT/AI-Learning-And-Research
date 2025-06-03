#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for Polynomial Regression tutorial.

This script demonstrates the complete analysis pipeline for polynomial regression,
including data generation, model training, evaluation, and visualization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from data import (
    generate_synthetic_data,
    load_real_data,
    preprocess_data,
    visualize_data,
    create_polynomial_features,
    generate_basis_functions_data
)

from model import (
    PolynomialRegressionFromScratch,
    PolynomialRegressionSklearn,
    select_optimal_degree,
    evaluate_regression_model,
    plot_learning_curves,
    plot_polynomial_predictions
)


def run_synthetic_data_example(n_samples=100, test_size=0.2, max_degree=10, noise=0.3, 
                               function_type='polynomial', learning_rate=0.01, n_iterations=1000,
                               regularization=None, alpha=0.1, visualize=True, random_state=42):
    """
    Run a complete pipeline with synthetic data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    test_size : float
        Proportion of the dataset to include in the test split
    max_degree : int
        Maximum polynomial degree to consider
    noise : float
        Noise level for synthetic data generation
    function_type : str
        Type of function to generate data from ('polynomial', 'sine', 'exponential')
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
    print(f"POLYNOMIAL REGRESSION WITH SYNTHETIC {function_type.upper()} DATA")
    print("="*80)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic {function_type} data with noise={noise}...")
    X, y, true_function = generate_synthetic_data(
        n_samples=n_samples, 
        noise=noise, 
        function_type=function_type, 
        random_state=random_state
    )
    
    # Visualize data
    if visualize:
        visualize_data(X, y, title=f"Synthetic {function_type.capitalize()} Data")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=False, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Select optimal polynomial degree using cross-validation
    print("\nSelecting optimal polynomial degree...")
    optimal_degree, cv_scores = select_optimal_degree(
        X_train, y_train, max_degree=max_degree, cv=5, scoring='neg_mean_squared_error'
    )
    print(f"Optimal polynomial degree: {optimal_degree}")
    
    if visualize:
        # Plot CV scores vs. polynomial degree
        plt.figure(figsize=(10, 6))
        plt.errorbar(range(1, max_degree + 1), 
                    [-score.mean() for score in cv_scores], 
                    yerr=[score.std() for score in cv_scores],
                    marker='o', linestyle='-')
        plt.axvline(x=optimal_degree, color='r', linestyle='--', 
                   label=f'Optimal Degree: {optimal_degree}')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error')
        plt.title('Cross-Validation Error vs. Polynomial Degree')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Train polynomial regression model from scratch
    print("\nTraining polynomial regression model from scratch...")
    model_scratch = PolynomialRegressionFromScratch(
        degree=optimal_degree,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization,
        alpha=alpha
    )
    model_scratch.fit(X_train, y_train)
    
    # Evaluate scratch model
    print("\nEvaluating model from scratch:")
    metrics_scratch = evaluate_regression_model(model_scratch, X_test, y_test, 
                                              model_name="Polynomial Regression (Scratch)")
    
    if visualize:
        # Plot learning curve for scratch model
        model_scratch.plot_cost_history()
    
    # Train scikit-learn model with optimal degree
    print("\nTraining scikit-learn polynomial regression model...")
    model_sklearn = PolynomialRegressionSklearn(
        degree=optimal_degree,
        regularization=regularization,
        alpha=alpha
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics_sklearn = evaluate_regression_model(model_sklearn, X_test, y_test, 
                                              model_name="Polynomial Regression (Scikit-learn)")
    
    if visualize:
        # Plot polynomial fits with different degrees
        degrees_to_plot = [1, 2, optimal_degree, max_degree]
        plot_polynomial_predictions(X, y, degrees_to_plot, 
                                   true_function=true_function if function_type != 'real' else None)
    
    return model_sklearn, metrics_sklearn


def run_real_data_example(dataset_name='boston', test_size=0.2, max_degree=5,
                        regularization=None, alpha=0.1, visualize=True, random_state=42):
    """
    Run a complete pipeline with a real dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('boston', 'diabetes', 'california')
    test_size : float
        Proportion of the dataset to include in the test split
    max_degree : int
        Maximum polynomial degree to consider
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
    print(f"POLYNOMIAL REGRESSION WITH {dataset_name.upper()} DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names = load_real_data(dataset_name)
    
    # For visualization, select just one feature if there are many
    if X.shape[1] > 1 and visualize:
        print("Multiple features detected. Using the most correlated feature for visualization.")
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        best_feature_idx = np.abs(correlations).argmax()
        print(f"Selected feature: {feature_names[best_feature_idx]} (correlation: {correlations[best_feature_idx]:.4f})")
        X_vis = X[:, [best_feature_idx]]
    else:
        X_vis = X
    
    if visualize:
        if X_vis.shape[1] == 1:  # Can only visualize 1D data easily
            visualize_data(X_vis, y, title=f"{dataset_name.title()} Dataset", xlabel=feature_names[best_feature_idx])
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Select optimal polynomial degree using cross-validation
    print("\nSelecting optimal polynomial degree...")
    optimal_degree, cv_scores = select_optimal_degree(
        X_train, y_train, max_degree=max_degree, cv=5, scoring='neg_mean_squared_error'
    )
    print(f"Optimal polynomial degree: {optimal_degree}")
    
    if visualize:
        # Plot CV scores vs. polynomial degree
        plt.figure(figsize=(10, 6))
        plt.errorbar(range(1, max_degree + 1), 
                    [-score.mean() for score in cv_scores], 
                    yerr=[score.std() for score in cv_scores],
                    marker='o', linestyle='-')
        plt.axvline(x=optimal_degree, color='r', linestyle='--', 
                   label=f'Optimal Degree: {optimal_degree}')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error')
        plt.title('Cross-Validation Error vs. Polynomial Degree')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Train scikit-learn model with optimal degree
    print("\nTraining scikit-learn polynomial regression model...")
    model_sklearn = PolynomialRegressionSklearn(
        degree=optimal_degree,
        regularization=regularization,
        alpha=alpha
    )
    model_sklearn.fit(X_train, y_train)
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics_sklearn = evaluate_regression_model(model_sklearn, X_test, y_test, 
                                              model_name="Polynomial Regression (Scikit-learn)")
    
    if visualize and X_vis.shape[1] == 1:
        # Plot polynomial fits with different degrees for the selected feature
        X_vis_train, X_vis_test, y_vis_train, y_vis_test, _, _ = preprocess_data(
            X_vis, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
        )
        
        degrees_to_plot = [1, 2, optimal_degree, max_degree]
        plot_polynomial_predictions(X_vis, y, degrees_to_plot, xlabel=feature_names[best_feature_idx])
    
    return model_sklearn, metrics_sklearn


def run_basis_functions_example(n_samples=100, test_size=0.2, basis_type='gaussian',
                              n_basis=10, regularization='l2', alpha=0.1, visualize=True, random_state=42):
    """
    Run a pipeline with basis function expansions (beyond simple polynomials).
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    test_size : float
        Proportion of the dataset to include in the test split
    basis_type : str
        Type of basis functions ('gaussian', 'sigmoid', 'fourier')
    n_basis : int
        Number of basis functions to use
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
    print(f"REGRESSION WITH {basis_type.upper()} BASIS FUNCTIONS")
    print("="*80)
    
    # Generate synthetic data with basis functions
    print(f"\nGenerating synthetic data with {basis_type} basis functions...")
    X, y, X_basis, centers = generate_basis_functions_data(
        n_samples=n_samples,
        basis_type=basis_type,
        n_basis=n_basis,
        noise=0.1,
        random_state=random_state
    )
    
    # Visualize data
    if visualize:
        visualize_data(X, y, title=f"Synthetic Data for {basis_type.capitalize()} Basis Functions")
        
        # Plot basis functions
        plt.figure(figsize=(12, 6))
        xx = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        if basis_type == 'gaussian':
            for i, center in enumerate(centers):
                width = 0.1
                plt.plot(xx, np.exp(-(xx - center)**2 / (2 * width**2)), 
                        label=f'Basis {i+1}' if i < 5 else None)
        elif basis_type == 'sigmoid':
            for i, center in enumerate(centers):
                width = 0.5
                plt.plot(xx, 1 / (1 + np.exp(-(xx - center) / width)), 
                        label=f'Basis {i+1}' if i < 5 else None)
        elif basis_type == 'fourier':
            for i, freq in enumerate(centers):
                plt.plot(xx, np.sin(freq * xx), 
                        label=f'sin({freq:.1f}x)' if i < 5 else None)
        
        plt.title(f"{basis_type.capitalize()} Basis Functions")
        plt.xlabel("x")
        plt.ylabel("Basis Function Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Preprocess data (note: we already have the basis functions expansion)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X_basis, y, test_size=test_size, standardize_X=False, standardize_y=False, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples with {X_train.shape[1]} basis functions")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train linear regression model on the basis functions
    print("\nTraining linear regression model on basis functions...")
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    
    if regularization == 'l2':
        model = Ridge(alpha=alpha)
    elif regularization == 'l1':
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model:")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    if visualize:
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(X[:, 0], y, color='blue', alpha=0.5, label='Data')
        
        # Create a sorted version of the original data for smooth line plotting
        X_sorted_idx = np.argsort(X[:, 0])
        X_sorted = X[X_sorted_idx]
        
        # Transform the sorted X to basis functions
        if basis_type == 'gaussian':
            X_sorted_basis = np.column_stack([
                np.exp(-(X_sorted - center)**2 / (2 * 0.1**2)) for center in centers
            ])
        elif basis_type == 'sigmoid':
            X_sorted_basis = np.column_stack([
                1 / (1 + np.exp(-(X_sorted - center) / 0.5)) for center in centers
            ])
        elif basis_type == 'fourier':
            X_sorted_basis = np.column_stack([
                np.sin(freq * X_sorted) for freq in centers
            ])
        
        # Make predictions
        y_sorted_pred = model.predict(X_sorted_basis)
        
        plt.plot(X_sorted, y_sorted_pred, 'r-', lw=2, label='Prediction')
        plt.title(f"Regression with {basis_type.capitalize()} Basis Functions")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return model, {'mse': mse, 'r2': r2}


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Polynomial Regression Tutorial')
    
    parser.add_argument('--data-type', type=str, default='synthetic',
                      choices=['synthetic', 'real', 'basis'],
                      help='Type of data to use (synthetic, real, basis)')
    
    parser.add_argument('--function-type', type=str, default='polynomial',
                      choices=['polynomial', 'sine', 'exponential'],
                      help='Type of function for synthetic data')
    
    parser.add_argument('--dataset', type=str, default='boston',
                      choices=['boston', 'diabetes', 'california'],
                      help='Dataset name for real data')
    
    parser.add_argument('--basis-type', type=str, default='gaussian',
                      choices=['gaussian', 'sigmoid', 'fourier'],
                      help='Type of basis functions')
    
    parser.add_argument('--n-samples', type=int, default=100,
                      help='Number of samples for synthetic data')
    
    parser.add_argument('--n-basis', type=int, default=10,
                      help='Number of basis functions')
    
    parser.add_argument('--noise', type=float, default=0.3,
                      help='Noise level for synthetic data')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--max-degree', type=int, default=10,
                      help='Maximum polynomial degree to consider')
    
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
            test_size=args.test_size,
            max_degree=args.max_degree,
            noise=args.noise,
            function_type=args.function_type,
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
            max_degree=args.max_degree,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.data_type == 'basis':
        run_basis_functions_example(
            n_samples=args.n_samples,
            test_size=args.test_size,
            basis_type=args.basis_type,
            n_basis=args.n_basis,
            regularization=args.regularization,
            alpha=args.alpha,
            visualize=args.visualize,
            random_state=args.random_state
        )
    
    print("\nPolynomial Regression Tutorial completed successfully!")


if __name__ == "__main__":
    main()
