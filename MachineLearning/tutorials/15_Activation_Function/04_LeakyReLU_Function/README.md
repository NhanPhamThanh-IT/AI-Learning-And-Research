# Leaky ReLU Activation Function

## Introduction
The Leaky Rectified Linear Unit (Leaky ReLU) activation function is a variant of the standard ReLU function. It was introduced to address the "Dying ReLU" problem, where neurons can become inactive and stop learning during training. Unlike standard ReLU, which outputs 0 for all negative inputs, Leaky ReLU allows a small, non-zero gradient, preventing neurons from completely shutting down.

## Formula
The formula for the Leaky ReLU function is:

$$ f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases} $$

Where:
- $ x $ is the input value.
- $ \alpha $ is a small, positive constant (typically a small value like 0.01). This constant determines the slope of the function for negative inputs. A common value is $ \alpha = 0.01 $.

## Characteristics
- **Addresses the Dying ReLU Problem:** The primary advantage of Leaky ReLU over standard ReLU is that it allows a small gradient ($ \alpha $) for negative inputs. This ensures that neurons can still receive gradient updates even when their input is negative, preventing them from becoming permanently inactive (dying).
- **Computationally Efficient:** Like standard ReLU, Leaky ReLU is computationally efficient, involving only a simple comparison and multiplication operation.
- **Non-linearity:** It is a non-linear function, necessary for learning complex patterns.
- **Gradient for Negative Inputs:** The small positive slope for negative inputs ensures that gradients can flow during backpropagation even for negative activations.
- **Potential for Improved Performance:** In some cases, using Leaky ReLU can lead to faster training and better model performance compared to using standard ReLU, especially in deep networks where the Dying ReLU problem might be more prominent.

## Applications
Leaky ReLU can be used as an alternative to standard ReLU in various neural network architectures, including CNNs and MLPs. It is particularly useful in scenarios where the Dying ReLU problem is observed or suspected. While not always guaranteed to outperform ReLU, it's often worth trying as a drop-in replacement.

## Comparison with ReLU
- **Negative Inputs:** ReLU outputs 0 for negative inputs ($ f(x)=0 $ for $ x \le 0 $), while Leaky ReLU outputs a small linear component ($ f(x)=\alpha x $ for $ x \le 0 $).
- **Gradient for Negative Inputs:** ReLU has a gradient of 0 for negative inputs, whereas Leaky ReLU has a small positive gradient ($ \alpha $).
- **Dying Neurons:** Leaky ReLU is less susceptible to the Dying ReLU problem than standard ReLU.

## Example
We can illustrate the Leaky ReLU function using Python. 