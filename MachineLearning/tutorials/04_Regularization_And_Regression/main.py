#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for Regularization in Regression tutorial.

This script demonstrates the complete analysis pipeline for regularized regression models,
including Ridge, Lasso, and Elastic Net regularization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import (
    generate_synthetic_data,
    load_real_data,
    preprocess_data,
    visualize_data,
    generate_multicollinear_data
)

from model import (
    RidgeRegressionFromScratch,
    LassoRegressionFromScratch,
    ElasticNetRegressionFromScratch,
    RidgeRegressionSklearn,
    LassoRegressionSklearn,
    ElasticNetRegressionSklearn,
    evaluate_regression_model,
    plot_regularization_path,
    plot_coefficient_comparison
)


def run_ridge_example(n_samples=100, n_features=20, test_size=0.2, noise=0.5, 
                    multicollinearity=True, learning_rate=0.01, n_iterations=1000,
                    alphas=None, visualize=True, random_state=42):
    """
    Run a complete pipeline for Ridge Regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Proportion of the dataset to include in the test split
    noise : float
        Noise level for synthetic data generation
    multicollinearity : bool
        Whether to generate data with multicollinearity
    learning_rate : float
        Learning rate for gradient descent (for scratch implementation)
    n_iterations : int
        Number of iterations for training (for scratch implementation)
    alphas : list or None
        List of alpha values for regularization path
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("RIDGE REGRESSION (L2 REGULARIZATION)")
    print("="*80)
    
    if alphas is None:
        alphas = np.logspace(-3, 3, 100)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data with {n_features} features {'' if not multicollinearity else 'with multicollinearity'}...")
    X, y, true_coef = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=noise,
        multicollinearity=multicollinearity,
        random_state=random_state
    )
    
    if visualize and n_features < 10:
        # Visualize correlation between features
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(X, rowvar=False)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=n_features < 10)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train Ridge model from scratch with cross-validation
    print("\nTraining Ridge Regression model from scratch...")
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_mse = float('inf')
    best_model_scratch = None
    
    for alpha in alpha_values:
        model_scratch = RidgeRegressionFromScratch(
            alpha=alpha,
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        model_scratch.fit(X_train, y_train)
        
        # Simple validation
        y_pred = model_scratch.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"  Alpha = {alpha:.4f}, Test MSE = {mse:.4f}")
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model_scratch = model_scratch
    
    print(f"\nBest alpha from scratch implementation: {best_alpha}")
    
    # Evaluate best scratch model
    print("\nEvaluating best scratch model:")
    metrics_scratch = evaluate_regression_model(best_model_scratch, X_test, y_test, 
                                             model_name="Ridge Regression (Scratch)")
    
    if visualize:
        # Plot cost history for best model
        best_model_scratch.plot_cost_history()
    
    # Train scikit-learn Ridge model with cross-validation
    print("\nTraining scikit-learn Ridge Regression model with cross-validation...")
    model_sklearn = RidgeRegressionSklearn(alphas=alphas, cv=5)
    model_sklearn.fit(X_train, y_train)
    
    print(f"Best alpha from scikit-learn: {model_sklearn.alpha_}")
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics_sklearn = evaluate_regression_model(model_sklearn, X_test, y_test, 
                                             model_name="Ridge Regression (Scikit-learn)")
    
    if visualize:
        # Plot regularization path
        plot_regularization_path(X_train, y_train, alphas, model_type='ridge')
        
        # Compare true coefficients with estimated coefficients
        if true_coef is not None:
            plot_coefficient_comparison(
                true_coef, 
                model_sklearn.coef_, 
                title="True vs. Ridge Estimated Coefficients",
                y_label_1="True Coefficients",
                y_label_2="Ridge Estimated Coefficients"
            )
    
    return model_sklearn, metrics_sklearn


def run_lasso_example(n_samples=100, n_features=20, test_size=0.2, noise=0.5, 
                    multicollinearity=True, sparse=True, learning_rate=0.01, n_iterations=5000,
                    alphas=None, visualize=True, random_state=42):
    """
    Run a complete pipeline for Lasso Regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Proportion of the dataset to include in the test split
    noise : float
        Noise level for synthetic data generation
    multicollinearity : bool
        Whether to generate data with multicollinearity
    sparse : bool
        Whether to generate sparse true coefficients
    learning_rate : float
        Learning rate for gradient descent (for scratch implementation)
    n_iterations : int
        Number of iterations for training (for scratch implementation)
    alphas : list or None
        List of alpha values for regularization path
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("LASSO REGRESSION (L1 REGULARIZATION)")
    print("="*80)
    
    if alphas is None:
        alphas = np.logspace(-3, 1, 100)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data with {n_features} features...")
    X, y, true_coef = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=noise,
        multicollinearity=multicollinearity,
        random_state=random_state
    )
    
    # Make coefficients sparse if requested
    if sparse and true_coef is not None:
        mask = np.random.rand(n_features) > 0.7
        true_coef = true_coef * mask
        # Regenerate y with sparse coefficients
        y = X.dot(true_coef) + noise * np.random.randn(n_samples)
    
    if visualize and true_coef is not None:
        # Visualize true coefficients
        plt.figure(figsize=(10, 6))
        plt.stem(range(n_features), true_coef)
        plt.title("True Coefficient Values")
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train Lasso model from scratch with cross-validation
    print("\nTraining Lasso Regression model from scratch...")
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_alpha = None
    best_mse = float('inf')
    best_model_scratch = None
    
    for alpha in alpha_values:
        model_scratch = LassoRegressionFromScratch(
            alpha=alpha,
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        model_scratch.fit(X_train, y_train)
        
        # Simple validation
        y_pred = model_scratch.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"  Alpha = {alpha:.4f}, Test MSE = {mse:.4f}")
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model_scratch = model_scratch
    
    print(f"\nBest alpha from scratch implementation: {best_alpha}")
    
    # Evaluate best scratch model
    print("\nEvaluating best scratch model:")
    metrics_scratch = evaluate_regression_model(best_model_scratch, X_test, y_test, 
                                             model_name="Lasso Regression (Scratch)")
    
    if visualize:
        # Plot cost history for best model
        best_model_scratch.plot_cost_history()
    
    # Train scikit-learn Lasso model with cross-validation
    print("\nTraining scikit-learn Lasso Regression model with cross-validation...")
    model_sklearn = LassoRegressionSklearn(alphas=alphas, cv=5)
    model_sklearn.fit(X_train, y_train)
    
    print(f"Best alpha from scikit-learn: {model_sklearn.alpha_}")
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics_sklearn = evaluate_regression_model(model_sklearn, X_test, y_test, 
                                             model_name="Lasso Regression (Scikit-learn)")
    
    if visualize:
        # Plot regularization path
        plot_regularization_path(X_train, y_train, alphas, model_type='lasso')
        
        # Compare true coefficients with estimated coefficients
        if true_coef is not None:
            plot_coefficient_comparison(
                true_coef, 
                model_sklearn.coef_, 
                title="True vs. Lasso Estimated Coefficients",
                y_label_1="True Coefficients",
                y_label_2="Lasso Estimated Coefficients"
            )
        
        # Plot sparsity (number of non-zero coefficients)
        coefs = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_train, y_train)
            coefs.append(np.sum(lasso.coef_ != 0))
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(alphas, coefs)
        plt.xlabel('Alpha')
        plt.ylabel('Number of Non-Zero Coefficients')
        plt.title('Lasso Model Sparsity')
        plt.axvline(model_sklearn.alpha_, color='r', linestyle='--', 
                   label=f'CV Alpha: {model_sklearn.alpha_:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return model_sklearn, metrics_sklearn


def run_elastic_net_example(n_samples=100, n_features=20, test_size=0.2, noise=0.5, 
                          multicollinearity=True, l1_ratio=0.5, alphas=None, 
                          visualize=True, random_state=42):
    """
    Run a complete pipeline for Elastic Net Regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Proportion of the dataset to include in the test split
    noise : float
        Noise level for synthetic data generation
    multicollinearity : bool
        Whether to generate data with multicollinearity
    l1_ratio : float
        Ratio of L1 penalty in the Elastic Net (0 = Ridge, 1 = Lasso)
    alphas : list or None
        List of alpha values for regularization path
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("ELASTIC NET REGRESSION (L1 + L2 REGULARIZATION)")
    print("="*80)
    
    if alphas is None:
        alphas = np.logspace(-3, 1, 100)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data with {n_features} features...")
    X, y, true_coef = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=noise,
        multicollinearity=multicollinearity,
        random_state=random_state
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train Elastic Net model from scratch
    print("\nTraining Elastic Net Regression model from scratch...")
    model_scratch = ElasticNetRegressionFromScratch(
        alpha=0.1,
        l1_ratio=l1_ratio,
        learning_rate=0.01,
        n_iterations=5000
    )
    model_scratch.fit(X_train, y_train)
    
    # Evaluate scratch model
    print("\nEvaluating scratch model:")
    metrics_scratch = evaluate_regression_model(model_scratch, X_test, y_test, 
                                             model_name="Elastic Net Regression (Scratch)")
    
    if visualize:
        # Plot cost history
        model_scratch.plot_cost_history()
    
    # Train scikit-learn Elastic Net model with cross-validation
    print("\nTraining scikit-learn Elastic Net Regression model with cross-validation...")
    model_sklearn = ElasticNetRegressionSklearn(alphas=alphas, l1_ratio=l1_ratio, cv=5)
    model_sklearn.fit(X_train, y_train)
    
    print(f"Best alpha from scikit-learn: {model_sklearn.alpha_}")
    
    # Evaluate scikit-learn model
    print("\nEvaluating scikit-learn model:")
    metrics_sklearn = evaluate_regression_model(model_sklearn, X_test, y_test, 
                                             model_name="Elastic Net Regression (Scikit-learn)")
    
    if visualize:
        # Compare true coefficients with estimated coefficients
        if true_coef is not None:
            plot_coefficient_comparison(
                true_coef, 
                model_sklearn.coef_, 
                title="True vs. Elastic Net Estimated Coefficients",
                y_label_1="True Coefficients",
                y_label_2="Elastic Net Estimated Coefficients"
            )
        
        # Compare models with different l1_ratios
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        plt.figure(figsize=(12, 10))
        
        for i, ratio in enumerate(l1_ratios):
            model = ElasticNetRegressionSklearn(alphas=alphas, l1_ratio=ratio, cv=5)
            model.fit(X_train, y_train)
            
            plt.subplot(len(l1_ratios), 1, i + 1)
            plt.stem(range(n_features), model.coef_)
            plt.title(f"Elastic Net Coefficients (l1_ratio={ratio})")
            plt.ylabel("Coefficient Value")
            if i == len(l1_ratios) - 1:
                plt.xlabel("Feature Index")
        
        plt.tight_layout()
        plt.show()
    
    return model_sklearn, metrics_sklearn


def run_real_data_example(dataset_name='boston', test_size=0.2, visualize=True, random_state=42):
    """
    Run a complete pipeline with a real dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('boston', 'diabetes', 'california')
    test_size : float
        Proportion of the dataset to include in the test split
    visualize : bool
        Whether to generate visualizations
    random_state : int
        Random seed for reproducibility
    """
    print("\n" + "="*80)
    print(f"REGULARIZATION COMPARISON WITH {dataset_name.upper()} DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names = load_real_data(dataset_name)
    
    print(f"Dataset shape: {X.shape} samples, {len(feature_names)} features")
    
    if visualize:
        # Display feature correlation heatmap
        plt.figure(figsize=(12, 10))
        df = pd.DataFrame(X, columns=feature_names)
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                  square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        plt.title(f'{dataset_name.title()} Dataset Feature Correlations')
        plt.tight_layout()
        plt.show()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        X, y, test_size=test_size, standardize_X=True, standardize_y=True, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train and evaluate models with different regularization methods
    models = {}
    metrics = {}
    
    print("\nTraining and evaluating models with different regularization methods...")
    
    # Linear Regression (no regularization)
    from sklearn.linear_model import LinearRegression
    print("\n1. Linear Regression (No Regularization)")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    metrics['Linear'] = evaluate_regression_model(lr, X_test, y_test, 
                                              model_name="Linear Regression")
    models['Linear'] = lr
    
    # Ridge Regression
    print("\n2. Ridge Regression (L2 Regularization)")
    ridge = RidgeRegressionSklearn(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge.fit(X_train, y_train)
    print(f"Best alpha: {ridge.alpha_}")
    metrics['Ridge'] = evaluate_regression_model(ridge, X_test, y_test, 
                                              model_name="Ridge Regression")
    models['Ridge'] = ridge
    
    # Lasso Regression
    print("\n3. Lasso Regression (L1 Regularization)")
    lasso = LassoRegressionSklearn(alphas=np.logspace(-5, 1, 100), cv=5)
    lasso.fit(X_train, y_train)
    print(f"Best alpha: {lasso.alpha_}")
    metrics['Lasso'] = evaluate_regression_model(lasso, X_test, y_test, 
                                              model_name="Lasso Regression")
    models['Lasso'] = lasso
    
    # Elastic Net Regression
    print("\n4. Elastic Net Regression (L1 + L2 Regularization)")
    elastic_net = ElasticNetRegressionSklearn(alphas=np.logspace(-5, 1, 100), l1_ratio=0.5, cv=5)
    elastic_net.fit(X_train, y_train)
    print(f"Best alpha: {elastic_net.alpha_}")
    metrics['ElasticNet'] = evaluate_regression_model(elastic_net, X_test, y_test, 
                                                  model_name="Elastic Net Regression")
    models['ElasticNet'] = elastic_net
    
    if visualize:
        # Compare model performance
        performance = {
            'MSE': [metrics[m]['mse'] for m in metrics],
            'R²': [metrics[m]['r2'] for m in metrics]
        }
        
        plt.figure(figsize=(12, 10))
        
        # MSE comparison
        plt.subplot(2, 1, 1)
        plt.bar(metrics.keys(), performance['MSE'])
        plt.title('Mean Squared Error Comparison')
        plt.ylabel('MSE (lower is better)')
        plt.grid(axis='y', alpha=0.3)
        
        # R² comparison
        plt.subplot(2, 1, 2)
        plt.bar(metrics.keys(), performance['R²'])
        plt.title('R² Score Comparison')
        plt.ylabel('R² (higher is better)')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Compare coefficients across models
        plt.figure(figsize=(14, 8))
        
        # Get all coefficients
        coefs = pd.DataFrame({
            'Linear': models['Linear'].coef_,
            'Ridge': models['Ridge'].coef_,
            'Lasso': models['Lasso'].coef_,
            'ElasticNet': models['ElasticNet'].coef_,
        }, index=feature_names)
        
        # Plot coefficients
        coefs.plot(kind='bar', figsize=(14, 8))
        plt.title('Model Coefficient Comparison')
        plt.ylabel('Coefficient Value')
        plt.xlabel('Feature')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper right')
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
    parser = argparse.ArgumentParser(description='Regularization in Regression Tutorial')
    
    parser.add_argument('--model-type', type=str, default='comparison',
                      choices=['ridge', 'lasso', 'elastic-net', 'comparison', 'real-data'],
                      help='Type of model to run')
    
    parser.add_argument('--dataset', type=str, default='boston',
                      choices=['boston', 'diabetes', 'california'],
                      help='Dataset name for real data')
    
    parser.add_argument('--n-samples', type=int, default=100,
                      help='Number of samples for synthetic data')
    
    parser.add_argument('--n-features', type=int, default=20,
                      help='Number of features for synthetic data')
    
    parser.add_argument('--noise', type=float, default=0.5,
                      help='Noise level for synthetic data')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--no-multicollinearity', dest='multicollinearity', action='store_false',
                      help='Disable multicollinearity in synthetic data')
    
    parser.add_argument('--l1-ratio', type=float, default=0.5,
                      help='L1 ratio for Elastic Net (0 = Ridge, 1 = Lasso)')
    
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                      help='Disable visualizations')
    
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.model_type == 'ridge':
        run_ridge_example(
            n_samples=args.n_samples,
            n_features=args.n_features,
            test_size=args.test_size,
            noise=args.noise,
            multicollinearity=args.multicollinearity,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.model_type == 'lasso':
        run_lasso_example(
            n_samples=args.n_samples,
            n_features=args.n_features,
            test_size=args.test_size,
            noise=args.noise,
            multicollinearity=args.multicollinearity,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.model_type == 'elastic-net':
        run_elastic_net_example(
            n_samples=args.n_samples,
            n_features=args.n_features,
            test_size=args.test_size,
            noise=args.noise,
            multicollinearity=args.multicollinearity,
            l1_ratio=args.l1_ratio,
            visualize=args.visualize,
            random_state=args.random_state
        )
        
    elif args.model_type == 'comparison':
        # Run all three models and compare
        print("\n" + "="*80)
        print("COMPARING DIFFERENT REGULARIZATION METHODS")
        print("="*80)
        
        # Generate synthetic data with multicollinearity
        print(f"\nGenerating synthetic data with {args.n_features} features...")
        X, y, true_coef = generate_synthetic_data(
            n_samples=args.n_samples, 
            n_features=args.n_features, 
            noise=args.noise,
            multicollinearity=args.multicollinearity,
            random_state=args.random_state
        )
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
            X, y, test_size=args.test_size, standardize_X=True, standardize_y=True, random_state=args.random_state
        )
        
        models = {}
        metrics = {}
        
        # Linear Regression (no regularization)
        from sklearn.linear_model import LinearRegression
        print("\n1. Linear Regression (No Regularization)")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        metrics['Linear'] = evaluate_regression_model(lr, X_test, y_test, 
                                                  model_name="Linear Regression")
        models['Linear'] = lr
        
        # Ridge Regression
        print("\n2. Ridge Regression (L2 Regularization)")
        ridge = RidgeRegressionSklearn(alphas=np.logspace(-3, 3, 100), cv=5)
        ridge.fit(X_train, y_train)
        print(f"Best alpha: {ridge.alpha_}")
        metrics['Ridge'] = evaluate_regression_model(ridge, X_test, y_test, 
                                                  model_name="Ridge Regression")
        models['Ridge'] = ridge
        
        # Lasso Regression
        print("\n3. Lasso Regression (L1 Regularization)")
        lasso = LassoRegressionSklearn(alphas=np.logspace(-5, 1, 100), cv=5)
        lasso.fit(X_train, y_train)
        print(f"Best alpha: {lasso.alpha_}")
        metrics['Lasso'] = evaluate_regression_model(lasso, X_test, y_test, 
                                                  model_name="Lasso Regression")
        models['Lasso'] = lasso
        
        # Elastic Net Regression
        print("\n4. Elastic Net Regression (L1 + L2 Regularization)")
        elastic_net = ElasticNetRegressionSklearn(alphas=np.logspace(-5, 1, 100), l1_ratio=args.l1_ratio, cv=5)
        elastic_net.fit(X_train, y_train)
        print(f"Best alpha: {elastic_net.alpha_}")
        metrics['ElasticNet'] = evaluate_regression_model(elastic_net, X_test, y_test, 
                                                      model_name="Elastic Net Regression")
        models['ElasticNet'] = elastic_net
        
        if args.visualize:
            # Visualize coefficient comparison
            plt.figure(figsize=(15, 10))
            
            if true_coef is not None:
                plt.subplot(5, 1, 1)
                plt.stem(range(len(true_coef)), true_coef)
                plt.title("True Coefficients")
                plt.ylabel("Value")
                plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 2)
            plt.stem(range(len(models['Linear'].coef_)), models['Linear'].coef_)
            plt.title("Linear Regression (No Regularization)")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 3)
            plt.stem(range(len(models['Ridge'].coef_)), models['Ridge'].coef_)
            plt.title(f"Ridge Regression (alpha={models['Ridge'].alpha_:.4f})")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 4)
            plt.stem(range(len(models['Lasso'].coef_)), models['Lasso'].coef_)
            plt.title(f"Lasso Regression (alpha={models['Lasso'].alpha_:.4f})")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 5)
            plt.stem(range(len(models['ElasticNet'].coef_)), models['ElasticNet'].coef_)
            plt.title(f"Elastic Net Regression (alpha={models['ElasticNet'].alpha_:.4f}, l1_ratio={args.l1_ratio})")
            plt.xlabel("Feature Index")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Compare model performance
            performance = {
                'MSE': [metrics[m]['mse'] for m in metrics],
                'R²': [metrics[m]['r2'] for m in metrics]
            }
            
            plt.figure(figsize=(12, 10))
            
            # MSE comparison
            plt.subplot(2, 1, 1)
            plt.bar(metrics.keys(), performance['MSE'])
            plt.title('Mean Squared Error Comparison')
            plt.ylabel('MSE (lower is better)')
            plt.grid(axis='y', alpha=0.3)
            
            # R² comparison
            plt.subplot(2, 1, 2)
            plt.bar(metrics.keys(), performance['R²'])
            plt.title('R² Score Comparison')
            plt.ylabel('R² (higher is better)')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
    elif args.model_type == 'real-data':
        run_real_data_example(
            dataset_name=args.dataset,
            test_size=args.test_size,
            visualize=args.visualize,
            random_state=args.random_state
        )
    
    print("\nRegularization in Regression Tutorial completed successfully!")


if __name__ == "__main__":
    main()
