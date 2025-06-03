# Softmax Activation Function

## Introduction
The Softmax function, also known as the normalized exponential function, is an activation function often used in the output layer of neural networks for multi-class classification problems. It takes a vector of arbitrary real numbers and transforms it into a probability distribution over the classes, meaning the output is a vector of values between 0 and 1 that sum up to 1. This makes it ideal for representing the confidence scores for each class in a classification task.

## Formula
Given an input vector $ \mathbf{z} = [z_1, z_2, \dots, z_K] $ (where $ K $ is the number of classes), the Softmax function is calculated as follows:

$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$

Where:
- $ z_i $ is the $ i $-th element of the input vector.
- $ K $ is the total number of classes.
- $ e $ is the base of the natural logarithm.

To prevent numerical instability (e.g., overflow or underflow) when dealing with very large or very small exponents, it is common to use the *stability trick* by subtracting the maximum value of $ \mathbf{z} $ from each element before exponentiating:

$$ \text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}} $$

## Characteristics
- **Conversion to Probabilities:** The Softmax function converts raw scores (logits) into a probability distribution over the classes. Each element of the output vector represents the probability of the input belonging to a particular class.
- **Sum of 1:** The sum of all elements in the Softmax output vector always equals 1. This property is essential for interpreting the output as probabilities.
- **Enhance Differences:** The exponential operation in Softmax exaggerates the differences between the input values. Larger input values will result in significantly higher corresponding probabilities, making the most likely class stand out.
- **Differentiable:** The Softmax function is differentiable, which is necessary for gradient-based optimization algorithms like backpropagation.

## Applications
The Softmax function is the standard choice for the output layer of neural networks trained for multi-class classification tasks. It is used in a wide range of applications, including image classification, natural language processing (e.g., predicting the next word in a sequence), and speech recognition.

## Relationship with Cross-Entropy Loss
Softmax is often paired with the cross-entropy loss function for training classification models. The combination of Softmax and cross-entropy loss provides a well-behaved gradient that simplifies the optimization process.

## Example
We can illustrate the Softmax function using Python. 