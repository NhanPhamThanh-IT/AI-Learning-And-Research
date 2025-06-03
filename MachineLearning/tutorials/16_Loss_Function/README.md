# Loss Functions in Machine Learning

## What is a Loss Function?
A **loss function** (also known as a cost function or objective function) is a mathematical function that measures the difference between the predicted values produced by a machine learning model and the actual target values from the data. The loss function quantifies how well or poorly the model's predictions match the true values.

## Why are Loss Functions Important?
Loss functions are at the core of training machine learning models. During training, algorithms use the loss function to guide the optimization process—adjusting model parameters to minimize the loss. The choice of loss function directly impacts how the model learns and what it prioritizes (e.g., penalizing large errors, focusing on probability calibration, etc.).

## Types of Loss Functions
Loss functions can be broadly categorized based on the type of machine learning task:

### 1. Regression Loss Functions
Used when the target variable is continuous (e.g., predicting house prices).
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions, without considering their direction. Less sensitive to outliers.
- **Mean Squared Error (MSE):** Measures the average of the squares of the errors. Penalizes larger errors more heavily, making it sensitive to outliers.

### 2. Classification Loss Functions
Used when the target variable is categorical (e.g., classifying emails as spam or not spam).
- **Cross Entropy Loss:** Measures the dissimilarity between the true labels and the predicted probabilities. Used for both binary and multi-class classification.

## Overview of Included Loss Functions
This directory contains detailed tutorials and examples for the following loss functions:

### 1. [Mean Absolute Error (MAE)](01_Mean_Absolute_Error/README.md)
- **Definition:** Average of the absolute differences between predicted and actual values.
- **Use Cases:** Robust regression, fair error weighting.
- **Advantages:** Robust to outliers, easy to interpret.
- **Disadvantages:** Not differentiable at zero, may converge slower.
- **[See details and Python example.](01_Mean_Absolute_Error/README.md)**

### 2. [Mean Squared Error (MSE)](02_Mean_Squared_Error/README.md)
- **Definition:** Average of the squared differences between predicted and actual values.
- **Use Cases:** Linear and polynomial regression, general regression tasks.
- **Advantages:** Penalizes large errors, differentiable, simple interpretation.
- **Disadvantages:** Sensitive to outliers, not robust.
- **[See details and Python example.](02_Mean_Squared_Error/README.md)**

### 3. [Cross Entropy Loss](03_Cross_Entropy_Loss/README.md)
- **Definition:** Measures the difference between true and predicted probability distributions.
- **Use Cases:** Logistic regression, neural networks, classification problems.
- **Advantages:** Probabilistic output, strong penalization of confident errors, widely supported.
- **Disadvantages:** Numerical instability (log(0)), not suitable for regression.
- **[See details and Python example.](03_Cross_Entropy_Loss/README.md)**

## How to Choose a Loss Function
- **Regression Tasks:** Use MAE if you want robustness to outliers, or MSE if you want to penalize large errors more.
- **Classification Tasks:** Use Cross Entropy Loss for probabilistic outputs and when working with neural networks or logistic regression.
- **Consider the Data:** If your data contains many outliers, prefer MAE. If you care more about large errors, use MSE. For classification, Cross Entropy is the standard.

## Further Reading and Examples
Each subfolder contains a detailed explanation and a Python example for the respective loss function. Explore them for mathematical details, use cases, advantages/disadvantages, and code implementations:
- [01_Mean_Absolute_Error](01_Mean_Absolute_Error/README.md)
- [02_Mean_Squared_Error](02_Mean_Squared_Error/README.md)
- [03_Cross_Entropy_Loss](03_Cross_Entropy_Loss/README.md)

---

**References:**
- [Wikipedia: Loss function](https://en.wikipedia.org/wiki/Loss_function)
- [Scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Deep Learning Book: Loss Functions](https://www.deeplearningbook.org/contents/ml.html)
