# Activation Functions in Neural Networks

## Introduction
Activation functions are fundamental components of artificial neural networks, residing within the neurons themselves. Their primary role is to introduce non-linearity into the network's output. In a biological neuron, an activation function determines whether a neuron should fire based on the input it receives. Similarly, in an artificial neuron, the activation function decides the output of that neuron or whether it should be 'activated'.

Mathematically, a neuron typically performs a linear operation (a weighted sum of inputs plus a bias) followed by a non-linear activation function. The output of a neuron can be represented as:

$$ \text{Output} = f\left(\sum_{i=1}^{n} w_i x_i + b\right) $$

Where:
- $ x_i $ are the inputs to the neuron.
- $ w_i $ are the weights associated with each input.
- $ b $ is the bias term.
- $ n $ is the number of inputs.
- $ f $ is the activation function.
- $ \sum_{i=1}^{n} w_i x_i + b $ is the linear combination of inputs and weights, often referred to as the 'pre-activation' or 'logit'.

## The Importance of Non-linearity
Without activation functions (or with only linear activation functions), a neural network, no matter how many layers it has, would only be capable of learning linear transformations. A stack of linear layers is equivalent to a single linear layer. This would severely limit the network's ability to model and learn from complex, non-linear relationships present in most real-world data (like images, audio, or text). Non-linear activation functions allow the network to create complex mappings from inputs to outputs, enabling it to learn intricate patterns and solve highly complex tasks.

## Placement of Activation Functions
Activation functions are typically placed after the linear transformation in each hidden layer. In the output layer, the choice of activation function depends on the type of problem:
- **Binary Classification:** Sigmoid function is often used to output a probability between 0 and 1.
- **Multi-class Classification:** Softmax function is used to output a probability distribution over multiple classes.
- **Regression:** Often, no activation function is used in the output layer (identity function), or sometimes a linear activation is explicitly defined, as the output can be any real value.

## Common Activation Functions
This directory provides detailed tutorials on some of the most common and important activation functions used in neural networks:

### [Sigmoid Function](./01_Sigmoid_Function/README.md)
- **Description:** S-shaped function that squashes input values between 0 and 1. Used historically in hidden layers and still used in output layers for binary classification.
- **Formula:** $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
- **Pros:** Outputs are probabilities (useful for binary classification output), differentiable.
- **Cons:** Prone to vanishing gradients, output is not zero-centered, computationally expensive.

### [Softmax Function](./02_Softmax_Function/README.md)
- **Description:** Converts a vector of raw scores (logits) into a probability distribution over multiple classes. The sum of output probabilities is 1.
- **Formula:** $$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$
- **Pros:** Outputs a valid probability distribution, useful for multi-class classification output layers.
- **Cons:** Can suffer from numerical stability issues (often mitigated with the stability trick).

### [ReLU Function](./03_ReLU_Function/README.md)
- **Description:** The Rectified Linear Unit is the most commonly used activation function in hidden layers today. It outputs the input directly if it's positive, otherwise, it outputs zero.
- **Formula:** $$ f(x) = \max(0, x) $$
- **Pros:** Computationally efficient, helps mitigate vanishing gradients for positive inputs, introduces sparsity.
- **Cons:** Suffers from the Dying ReLU problem (neurons can become inactive).

### [Leaky ReLU Function](./04_LeakyReLU_Function/README.md)
- **Description:** Aims to solve the Dying ReLU problem by allowing a small, non-zero gradient for negative inputs.
- **Formula:** $$ f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases} $$
- **Pros:** Addresses Dying ReLU, computationally efficient, can sometimes lead to better performance than ReLU.
- **Cons:** Performance is not always guaranteed to be better than ReLU, the slope $ \alpha $ needs to be chosen (though often a small constant like 0.01 is used).

## Other Activation Functions
While the above are the most common, other activation functions exist, each with its own characteristics:
- **Tanh (Hyperbolic Tangent):** Squashes inputs to a range between -1 and 1. Zero-centered output, but still suffers from vanishing gradients.
  Formula: $$ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
- **ELU (Exponential Linear Unit):** Aims to make the mean activations closer to zero and provides non-zero gradients for negative inputs. Can lead to faster learning.
- **Swish (or SiLU):** A smooth, non-monotonic function that has shown promising results in deeper models.
  Formula: $$ \text{Swish}(x) = x \cdot \sigma(x) $$
- **GELU (Gaussian Error Linear Unit):** Currently the default activation in models like Transformers. It involves multiplying the input by the cumulative distribution function of the standard Gaussian distribution.

## Choosing an Activation Function
The choice of activation function depends on the specific task, network architecture, and dataset. ReLU is often the default choice for hidden layers due to its efficiency and performance, but variants like Leaky ReLU, ELU, or Swish might perform better in certain scenarios. Sigmoid and Softmax are typically reserved for output layers in classification tasks.

Experimentation is often required to find the best activation function for a given problem.

Understanding the mathematical properties and practical implications of different activation functions is key to building effective neural networks.
