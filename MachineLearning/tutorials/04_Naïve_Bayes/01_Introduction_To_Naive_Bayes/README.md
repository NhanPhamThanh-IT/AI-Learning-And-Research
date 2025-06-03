# Introduction to Naive Bayes

This directory contains a basic implementation and example of the Naive Bayes classification algorithm.

## What is Naive Bayes?

Naive Bayes is a classification algorithm that uses probability to predict which category a data point belongs to, assuming that all features are unrelated. It's based on Bayes' Theorem and is commonly used in high-dimensional text classification.

## Key Features

*   Simple probabilistic classifier with few parameters.
*   Predicts at a faster speed than many other classification algorithms.
*   Assumes features are independent of each other.
*   Used in applications like spam filtration, sentiment analysis, and document classification.

## Why "Naive" and "Bayes"?

*   **"Naive"**: Refers to the assumption that the presence of one feature does not affect other features (conditional independence).
*   **"Bayes"**: Refers to its foundation in Bayes' Theorem.

## Assumptions of Naive Bayes

*   **Feature independence:** Features are assumed to be independent of each other given the class label.
*   **Continuous features are normally distributed:** If features are continuous, they are often assumed to follow a Gaussian distribution within each class.
*   **Discrete features have multinomial distributions:** If features are discrete, they are assumed to have a multinomial distribution within each class.
*   **Features are equally important:** All features are assumed to contribute equally to the prediction.
*   **No missing data:** The data should ideally not contain missing values.

## Introduction to Bayes' Theorem

Bayes' Theorem provides a way to reverse conditional probabilities:

$$ P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)} $$

Where:
*   $ P(y|X) $: Posterior probability of class $ y $ given features $ X $.
*   $ P(X|y) $: Likelihood, probability of features $ X $ given class $ y $.
*   $ P(y) $: Prior probability of class $ y $.
*   $ P(X) $: Marginal likelihood or evidence.

## How Naive Bayes Works

Given a feature vector $ X = (x_1, x_2, ..., x_n) $ and a class label $ y $, the goal is to compute $ P(y|X) $. Using Bayes' Theorem:

$$ P(y|x_1, ..., x_n) = \frac{P(x_1, ..., x_n | y) \cdot P(y)}{P(x_1, ..., x_n)} $$

The "naive" assumption of feature independence given the class allows us to simplify the likelihood term:

$$ P(x_1, ..., x_n | y) = P(x_1 | y) \cdot P(x_2 | y) \cdots P(x_n | y) = \prod_{i=1}^{n} P(x_i | y) $$

Substituting this back into Bayes' Theorem:

$$ P(y|x_1, ..., x_n) = \frac{P(y) \cdot \prod_{i=1}^{n} P(x_i | y)}{P(x_1, ..., x_n)} $$

Since the denominator $ P(x_1, ..., x_n) $ is constant for a given input, we can compare the probabilities for different classes based on the numerator:

$$ P(y|x_1, ..., x_n) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i | y) $$

The Naive Bayes classifier predicts the class $ \hat{y} $ that maximizes this probability:

$$ \hat{y} = \arg\max_{y} P(y) \cdot \prod_{i=1}^{n} P(x_i | y) $$

During the training phase, the model calculates the class priors $ P(y) $ and the conditional probabilities $ P(x_i | y) $ from the training data.

## Example Dataset

This directory uses a simple categorical dataset similar to the golf example described in the GeeksforGeeks article. The dataset includes features like Outlook, Temperature, Humidity, and Windy, and the goal is to predict if golf will be played (`Yes` or `No`). The `data.py` file contains the implementation for creating and splitting this dataset.

## Advantages of Naive Bayes

*   Easy to implement and computationally efficient.
*   Effective in cases with a large number of features.
*   Performs well even with limited training data.
*   Works well with categorical features.

## Disadvantages of Naive Bayes

*   Assumes feature independence, which is often not true in real-world data.
*   Can be sensitive to irrelevant attributes.
*   May assign zero probability to unseen events (can be mitigated with smoothing techniques like Laplace smoothing).

## Applications of Naive Bayes

*   Spam Email Filtering
*   Text Classification (Sentiment Analysis, Document Categorization)
*   Medical Diagnosis
*   Credit Scoring
*   Weather Prediction
