# Multiple Regression Tutorial

This directory contains a comprehensive tutorial on Multiple Linear Regression, including theory, implementation, and practical applications.

## Overview

Multiple Linear Regression is an extension of simple linear regression that allows us to model the relationship between a dependent variable and multiple independent variables. This tutorial provides an in-depth exploration of multiple regression concepts, implementation approaches, and practical applications.

## Contents

1. **Jupyter Notebook**: `02_Multiple_Regression.ipynb`
   - Comprehensive theoretical explanation
   - Step-by-step implementation examples
   - Visualizations and model evaluation

2. **Python Modules**:
   - `data.py`: Data generation, loading, and preprocessing functions
   - `model.py`: Multiple regression model implementation
   - `main.py`: Entry point for running the complete analysis pipeline

## Learning Objectives

By completing this tutorial, you will:
- Understand the mathematical foundation of multiple linear regression
- Learn how to implement multiple regression from scratch
- Know how to use scikit-learn for efficient implementation
- Be able to evaluate and interpret multiple regression models
- Gain practical experience with feature selection and feature engineering
- Understand the issues of multicollinearity and how to address them

## Key Concepts

1. **Multiple Regression Formula**:
   $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n + \varepsilon$$

2. **Assumptions**:
   - Linearity of the relationship between dependent and independent variables
   - Independence of the observations
   - Homoscedasticity (constant variance) of the residuals
   - Normality of the residual distribution
   - No multicollinearity between independent variables

3. **Evaluation Metrics**:
   - R-squared and Adjusted R-squared
   - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - F-statistic and p-values

## Prerequisites

- Basic understanding of Python programming
- Familiarity with NumPy and Matplotlib
- Knowledge of simple linear regression concepts
- Elementary understanding of statistics

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis.
