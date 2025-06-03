# Simple Logistic Regression

## Introduction

Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many complex extensions exist. In regression analysis, logistic regression is estimating the parameters of a logistic model (a form of binary regression). Mathematically, a binary logistic model has a dependent variable with two possible values, such that the expected value of the dependent variable is a Bernoulli distribution.

This tutorial covers the basic Simple Logistic Regression, where we predict a binary outcome based on a single independent variable.

## How it Works

The core of Logistic Regression is the sigmoid function (also known as the logistic function):

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

This function takes any real-valued number and maps it into a value between 0 and 1. This output can be interpreted as the probability of the instance belonging to a certain class.

The linear combination of input features and weights is fed into the sigmoid function:

$$ z = w^T x + b $$

where:
- $ w $ is the weight vector
- $ x $ is the input feature vector
- $ b $ is the bias term

The predicted probability is then:

$$ P(y=1|x) = \sigma(w^T x + b) $$

For binary classification, if $ P(y=1|x) \ge 0.5 $, we predict class 1, otherwise we predict class 0.

## Loss Function

The most common loss function for Logistic Regression is the Binary Cross-Entropy Loss (also known as Log Loss):

$$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] $$

where:
- $ y_i $ is the true label for instance $ i $
- $ \hat{y}_i $ is the predicted probability for instance $ i $
- $ N $ is the number of instances

This loss function penalizes confident wrong predictions more heavily.

## Optimization

The weights $ w $ and bias $ b $ are learned by minimizing the loss function, typically using gradient descent or similar optimization algorithms.

## Files in this directory

- `README.md`: This explanation.
- `data.py`: Generates synthetic data for a binary classification problem.
- `model.py`: Contains the implementation of the Simple Logistic Regression model.
- `main.py`: Demonstrates how to use the data and model files to train and evaluate the model.

## Getting Started

To run the example:

1. Make sure you have the necessary libraries installed (e.g., `numpy`, `sklearn`). You might need to create a `requirements.txt` file and install them using `pip install -r requirements.txt`.
2. Run the `main.py` file:

   ```bash
   python main.py
   ```

The `main.py` script will generate data, train the logistic regression model, and print the accuracy. 