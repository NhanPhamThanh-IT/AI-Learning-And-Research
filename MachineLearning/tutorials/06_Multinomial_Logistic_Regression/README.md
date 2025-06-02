# Multinomial Logistic Regression Tutorial

This directory contains a comprehensive tutorial on Multinomial Logistic Regression for multi-class classification problems.

## Overview

Multinomial Logistic Regression (also known as Softmax Regression) is an extension of binary logistic regression to multi-class classification problems. It allows us to predict probabilities for more than two classes. This tutorial provides an in-depth exploration of multinomial logistic regression concepts, implementation approaches, evaluation metrics, and practical applications.

## Contents

1. **Jupyter Notebook**: `06_Multinomial_Logistic_Regression.ipynb`
   - Comprehensive theoretical explanation
   - Step-by-step implementation examples
   - Visualizations and model evaluation

2. **Python Modules**:
   - `data.py`: Data generation, loading, and preprocessing functions
   - `model.py`: Multinomial logistic regression model implementation
   - `main.py`: Entry point for running the complete analysis pipeline

## Learning Objectives

By completing this tutorial, you will:
- Understand the mathematical foundation of multinomial logistic regression
- Learn how the softmax function extends logistic regression to multiple classes
- Implement multinomial logistic regression from scratch
- Use scikit-learn for efficient implementation
- Evaluate multi-class classification models using appropriate metrics
- Apply regularization techniques to prevent overfitting
- Understand the one-vs-rest and softmax approaches

## Key Concepts

1. **Softmax Function**:
   - Generalizes logistic function to multiple dimensions
   - Maps a vector of K real numbers to a probability distribution of K classes
   - P(y=k|x) = exp(β₀ᵏ + β₁ᵏx₁ + ... + βₙᵏxₙ) / ∑ₖ₌₁ᴷ exp(β₀ᵏ + β₁ᵏx₁ + ... + βₙᵏxₙ)

2. **Model Representation**:
   - Each class has its own set of model parameters
   - For K classes, there are K weight vectors
   - One class is often chosen as reference to avoid redundancy

3. **Cross-Entropy Loss**:
   - J(θ) = -1/m ∑ᵢ₌₁ᵐ ∑ₖ₌₁ᴷ y_ik * log(p_ik)
   - Where y_ik = 1 if sample i belongs to class k, else 0
   - p_ik is the predicted probability of sample i belonging to class k

4. **Implementation Approaches**:
   - One-vs-Rest (OvR): Train K binary classifiers
   - Multinomial: Direct optimization using softmax function

5. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score (macro, micro, weighted)
   - Confusion Matrix
   - Multi-class ROC and AUC

## Prerequisites

- Understanding of binary logistic regression
- Familiarity with gradient descent optimization
- Knowledge of evaluation metrics for classification
- Basic understanding of probability and statistics

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
