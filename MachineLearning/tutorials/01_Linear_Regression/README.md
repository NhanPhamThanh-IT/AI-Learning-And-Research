# Linear Regression Tutorials

This directory provides a comprehensive set of tutorials on Linear Regression, covering theoretical foundations, practical implementations, and hands-on exercises. The tutorials are organized into three main sections:

- **Simple Linear Regression**
- **Multiple Linear Regression**
- **Polynomial Regression**

Each section includes Python modules for data handling, model implementation, and running experiments, as well as detailed explanations and references.

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

Linear Regression is a fundamental machine learning technique for modeling the relationship between a dependent variable and one or more independent variables. This tutorial series covers:

- **Simple Linear Regression**: Modeling with a single feature.
- **Multiple Linear Regression**: Extending to multiple features.
- **Polynomial Regression**: Modeling non-linear relationships by introducing polynomial terms.

Each section provides both theoretical background and practical implementation, including code from scratch and usage of popular libraries such as NumPy and Matplotlib.

---

## Directory Structure

```
01_Linear_Regression/
  ├── 01_Simple_Linear_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  ├── 02_Multiple_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  ├── 03_Polynomial_Regression/
  │     ├── data.py
  │     ├── model.py
  │     ├── main.py
  │     └── README.md
  └── README.md (this file)
```

Each subdirectory contains:
- `data.py`: Data generation, loading, preprocessing, and visualization functions.
- `model.py`: Model implementation (from scratch and/or using libraries).
- `main.py`: Entry point for running experiments and analysis.
- `README.md`: Section-specific documentation.

---

## Learning Objectives

By working through these tutorials, you will:
- Understand the mathematical foundations of linear, multiple, and polynomial regression.
- Learn to implement regression models from scratch in Python.
- Gain experience with data preprocessing, feature engineering, and model evaluation.
- Explore the assumptions, strengths, and limitations of each regression technique.
- Learn to visualize data, model fits, and residuals.
- Understand how to prevent overfitting and select model complexity (especially for polynomial regression).

---

## Key Concepts

### Simple Linear Regression
- **Model**: $Y = \beta_0 + \beta_1 X + \varepsilon$
- **Parameter Estimation**: Ordinary Least Squares (OLS), Gradient Descent
- **Cost Function**: Mean Squared Error (MSE)
- **Assumptions**: Linearity, independence, homoscedasticity, normality
- **Evaluation**: R-squared, MSE, MAE

### Multiple Linear Regression
- **Model**: $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \varepsilon$
- **Feature Selection**: Handling multicollinearity, feature importance
- **Evaluation**: Adjusted R-squared, F-statistic, p-values

### Polynomial Regression
- **Model**: $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_n X^n + \varepsilon$
- **Feature Transformation**: Creating polynomial features
- **Model Selection**: Cross-validation, bias-variance trade-off
- **Overfitting & Regularization**: Techniques to control model complexity

---

## Prerequisites

- Basic Python programming
- Familiarity with NumPy and Matplotlib
- Elementary statistics and calculus
- Basic understanding of machine learning concepts

---

## How to Use

1. **Read the section-specific README.md** for theoretical background and instructions.
2. **Explore the Python modules** (`data.py`, `model.py`, `main.py`) to understand and run the code.
3. **Run `main.py`** in each section to execute the full analysis pipeline:
   - Data generation/loading
   - Model training and evaluation
   - Visualization of results
4. **Modify the code** to experiment with different datasets, parameters, or model settings.

---

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis*.
- [Scikit-learn Linear Models Documentation](https://scikit-learn.org/stable/modules/linear_model.html)

---

For more details, see the README.md in each subdirectory and the code comments throughout the modules.
