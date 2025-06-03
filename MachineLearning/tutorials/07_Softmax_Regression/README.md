# Softmax Regression

## Introduction
Softmax Regression, also known as Multinomial Logistic Regression, is a generalization of logistic regression to multi-class classification problems. It is widely used in machine learning for tasks where the target variable can take on more than two categories. Unlike binary logistic regression, which predicts the probability of a single class, softmax regression outputs a probability distribution over multiple classes.

## Theoretical Background
Given an input vector \( \mathbf{x} \) and a set of classes \( K \), softmax regression models the probability that \( \mathbf{x} \) belongs to class \( k \) as:

\[
P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}
\]

where:
- \( \mathbf{w}_k \) is the weight vector for class \( k \)
- \( b_k \) is the bias for class \( k \)
- \( K \) is the number of classes

The model outputs a probability distribution over all possible classes for each input. The predicted class is the one with the highest probability.

### Decision Boundaries
Softmax regression creates linear decision boundaries between classes. For each pair of classes, the boundary is defined by the set of points where the predicted probabilities are equal.

## Loss Function
The most common loss function for softmax regression is the cross-entropy loss:

\[
L = -\sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log P(y_i = k \mid \mathbf{x}_i)
\]

where \( y_{i,k} \) is 1 if sample \( i \) belongs to class \( k \), and 0 otherwise. This loss encourages the model to assign high probability to the correct class.

## Training
The parameters (weights and biases) are typically learned using gradient descent or its variants (such as stochastic gradient descent, Adam, etc.). The gradients are computed with respect to the cross-entropy loss. Regularization (L1 or L2) is often added to prevent overfitting.

### Regularization
- **L2 Regularization (Ridge)**: Adds a penalty proportional to the square of the weights.
- **L1 Regularization (Lasso)**: Adds a penalty proportional to the absolute value of the weights.

## Example
Suppose we have a dataset of handwritten digits (0-9). Softmax regression can be used to classify each image into one of the 10 classes. The model will output a probability for each digit, and the digit with the highest probability is chosen as the prediction.

## Applications
- Handwritten digit recognition (e.g., MNIST dataset)
- Document and text classification
- Image classification
- Medical diagnosis (multi-class problems)

## Advantages
- Simple and interpretable
- Efficient for small to medium-sized datasets
- Outputs calibrated probabilities

## Limitations
- Assumes linear decision boundaries between classes
- Not suitable for highly non-linear problems without feature engineering
- Performance may degrade with highly imbalanced classes

## Comparison with Other Methods
- **Logistic Regression**: Only for binary classification; softmax regression generalizes it to multi-class.
- **Decision Trees/Random Forests**: Can model non-linear relationships and interactions between features.
- **Neural Networks**: Can capture complex, non-linear patterns but require more data and computational resources.

## Practical Considerations
- Feature scaling can improve convergence speed.
- Regularization is important to avoid overfitting.
- For large numbers of classes, computational cost increases.

## References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press. 