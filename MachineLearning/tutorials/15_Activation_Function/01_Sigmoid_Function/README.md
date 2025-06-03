# Sigmoid Activation Function

## Introduction
The Sigmoid function, also known as the logistic function, is a common activation function in traditional neural networks, especially in the output layer for binary classification problems. It maps any real-valued number to a value between 0 and 1, making it suitable for output layers where the output is a probability.

## Formula
The formula for the Sigmoid function is:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Where:
- $ x $ is the input value (e.g., the weighted sum of inputs plus bias for a neuron).
- $ e $ is the base of the natural logarithm (approximately 2.71828).

## Characteristics
- **Output Range:** The output of the Sigmoid function is always between 0 and 1. This property makes it useful for output layers in binary classification to represent probabilities.
- **Monotonic:** The function is monotonically increasing.
- **Differentiable:** It is differentiable, which is crucial for backpropagation algorithms.
- **Derivative:** The derivative of the Sigmoid function can be expressed in terms of the function itself:
  $$ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $$
  The maximum value of the derivative is 0.25, occurring at $x=0$.
- **Vanishing Gradient Problem:** For very large positive or negative input values $ x $, the derivative of the Sigmoid function approaches 0. During backpropagation, gradients are multiplied. If the gradients in many layers are close to 0, the gradients flowing back to earlier layers become very small, effectively preventing the weights in those layers from being updated. This is known as the vanishing gradient problem, which makes it difficult to train deep networks with Sigmoid activation in hidden layers.

## Applications
The Sigmoid function was historically popular in hidden layers but is now primarily used in the output layer of neural networks for binary classification tasks, where the output needs to represent a probability.

## Limitations
- **Vanishing Gradients:** As discussed, this hinders the training of deep networks.
- **Outputs Not Zero-Centered:** The outputs are always positive (between 0 and 1). This can cause issues during training as gradients for the weights in the next layer will always have the same sign, potentially leading to zig-zagging optimization paths.
- **Computationally Expensive:** The `exp()` operation is more computationally expensive than simpler functions like ReLU.

## Example
We can illustrate the Sigmoid function using Python. 