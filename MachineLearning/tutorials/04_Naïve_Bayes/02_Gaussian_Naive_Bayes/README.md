# Gaussian Naive Bayes

This directory contains a basic implementation and example of the Gaussian Naive Bayes classification algorithm.

## What is Gaussian Naive Bayes?

Gaussian Naive Bayes is a variant of the Naive Bayes algorithm used for classification tasks where the continuous features are assumed to follow a Gaussian (Normal) distribution within each class. It is widely used due to its simplicity, efficiency, and good performance even with relatively small datasets.

## Mathematics Behind Gaussian Naive Bayes

The core assumption of Gaussian Naive Bayes is that the likelihood of a feature $ x_i $ given a class $ y_k $ follows a Gaussian Distribution. The Probability Density Function (PDF) of a Gaussian distribution is given by:

$$ P(x_i|y_k) = \frac{1}{\sigma_{ik} \sqrt{2\pi}} e^{-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}} $$

Where:
*   $ x_i $ is the value of the $ i $-th feature.
*   $ \mu_{ik} $ is the mean of the $ i $-th feature for class $ y_k $.
*   $ \sigma_{ik} $ is the standard deviation of the $ i $-th feature for class $ y_k $.
*   $ \pi $ is approximately 3.14159.
*   $ e $ is the base of the natural logarithm.

To classify a new data point $ X = (x_1, x_2, ..., x_n) $, the algorithm calculates the posterior probability for each class $ y_k $ using Bayes' Theorem and the naive independence assumption:

$$ P(y_k|X) \propto P(y_k) \cdot \prod_{i=1}^{n} P(x_i | y_k) $$

Using the Gaussian PDF for $ P(x_i | y_k) $:

$$ P(y_k|X) \propto P(y_k) \cdot \prod_{i=1}^{n} \frac{1}{\sigma_{ik} \sqrt{2\pi}} e^{-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}} $$

The predicted class $ \hat{y} $ is the one that maximizes this posterior probability:

$$ \hat{y} = \arg\max_{y_k} P(y_k) \cdot \prod_{i=1}^{n} P(x_i | y_k) $$

## Why it Works Well for Continuous Data

Gaussian Naive Bayes is effective for continuous data when the assumption of Gaussian distribution holds true. By modeling each feature's distribution within each class, it can effectively calculate the likelihoods needed for classification.

## Practical Example

The directory includes a simple example demonstrating Gaussian Naive Bayes, similar to the petal length example in the linked article. This involves calculating the mean and variance for each feature within each class from the training data and then using these parameters with the Gaussian PDF to make predictions on new data points.

## Advantages of Gaussian Naive Bayes

*   Easy to implement.
*   Computationally efficient.
*   Performs well even with limited training data.
*   Handles continuous features naturally by assuming a Gaussian distribution.

## Disadvantages of Gaussian Naive Bayes

*   The strong independence assumption between features may not hold in real-world data.
*   Performance can degrade if the continuous features do not follow a Gaussian distribution.

## Applications of Gaussian Naive Bayes

Similar to the general Naive Bayes, Gaussian Naive Bayes is used in various applications involving continuous features, such as:

*   Medical Diagnosis
*   Credit Scoring
*   Classification tasks with numerical data
