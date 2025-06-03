# Logistic Regression Tutorials

This directory contains tutorials on Logistic Regression, a fundamental classification algorithm in machine learning. It covers the extension of binary logistic regression to multi-class problems, specifically Multinomial and Softmax Regression.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Learning Objectives](#learning-objectives)
- [Key Concepts](#key-concepts)
- [Prerequisites](#prerequisites)
- [How to Use](#how-to-use)
- [References](#references)

---

## Overview

Logistic Regression is used for binary classification tasks, modeling the probability of an instance belonging to a particular class. This tutorial series primarily focuses on its extension to multi-class classification:

- **Multinomial Logistic Regression (Softmax Regression)**: Handling classification problems with more than two classes.

*Note: The directory `01_Logistic_Regression` is currently empty.* The main content for multi-class logistic regression is found in the `02_Multinomial_Logistic_Regression` and `03_Softmax_Regression` subdirectories, which cover similar concepts and implementations.

---

## Directory Structure

```
05_Logistic_Regression/
  ├── 01_Logistic_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  ├── 02_Multinomial_Logistic_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  ├── 03_Softmax_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  └── README.md (this file)
```

Each populated subdirectory contains:
- `data.py`: Functions for data generation, loading, preprocessing, and sometimes visualization.
- `model.py`: Implementation of the logistic/softmax regression model.
- `main.py`: Script to run the complete analysis pipeline.
- `README.md`: Section-specific documentation with theoretical details and usage.
- Jupyter Notebook (`.ipynb`): Interactive exploration and step-by-step tutorial.

---

## Learning Objectives

By working through these tutorials, you will:
- Understand the principles of logistic regression for classification.
- Learn how to extend logistic regression to handle multiple classes using the Softmax function.
- Understand and implement the Cross-Entropy loss function.
- Gain experience with gradient descent for model training.
- Learn about regularization techniques (L1, L2) to prevent overfitting.
- Understand and apply appropriate evaluation metrics for multi-class classification (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Explore different implementation approaches like One-vs-Rest and the direct Multinomial approach.

---

## Key Concepts

### Multinomial/Softmax Regression
- **Softmax Function**: Transforms raw scores into a probability distribution over multiple classes.
  \\[ P(y = k \mid \\mathbf{x}) = \\frac{\exp(\\mathbf{w}_k^T \\mathbf{x} + b_k)}{\\sum_{j=1}^K \\exp(\\mathbf{w}_j^T \\mathbf{x} + b_j)} \\]
- **Model Representation**: Each class has a weight vector and bias.
- **Cross-Entropy Loss**: Measures the performance of a classification model whose output is a probability value between 0 and 1.
  \\[ L = -\\sum_{i=1}^N \\sum_{k=1}^K y_{i,k} \\log P(y_i = k \mid \\mathbf{x}_i) \\]
- **Training**: Parameter optimization using gradient descent.
- **Regularization**: L1 and L2 penalties to control model complexity and prevent overfitting.
- **Implementation Approaches**: One-vs-Rest (OvR) vs. direct Multinomial optimization.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Decision Boundaries**: Linear boundaries are formed between classes.

---

## Prerequisites

- Basic Python programming.
- Familiarity with NumPy and potentially Matplotlib.
- Understanding of binary logistic regression concepts.
- Elementary knowledge of probability, statistics, and calculus.
- Basic understanding of machine learning concepts and evaluation metrics.

---

## How to Use

1. **Navigate to the subdirectories** `02_Multinomial_Logistic_Regression` and `03_Softmax_Regression`.
2. **Read the section-specific README.md** files for detailed theoretical explanations and instructions.
3. **Explore the Python modules** (`data.py`, `model.py`, `main.py`) to understand the implementation details.
4. **Run the `main.py` scripts** to execute the analysis pipelines, including data handling, model training, and evaluation.
5. **Work through the Jupyter Notebook** (`06_Multinomial_Logistic_Regression.ipynb`) for an interactive learning experience.
6. **Modify the code** to experiment with different datasets, parameters, or model variations.

---

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

---

For more details, see the README.md in each subdirectory and the code comments throughout the modules.
