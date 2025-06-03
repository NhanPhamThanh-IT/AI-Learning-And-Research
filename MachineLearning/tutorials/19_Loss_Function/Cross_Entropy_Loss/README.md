# Cross Entropy Loss Function

## Definition
Cross Entropy Loss is a crucial loss function used mainly for classification tasks. It measures the dissimilarity between the true probability distribution (actual labels) and the predicted probability distribution (model outputs). Lower cross entropy indicates that the predicted probabilities are closer to the true labels.

## Mathematical Formula
### Binary Cross Entropy (for binary classification)
For $n$ samples:

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

Where:
- $y_i$: Actual label for the $i$-th sample ($0$ or $1$)
- $p_i$: Predicted probability that the $i$-th sample belongs to class $1$
- $n$: Number of samples

### Categorical Cross Entropy (for multi-class classification)
For $n$ samples and $C$ classes:

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$

Where:
- $y_{i,c}$: Actual label (one-hot encoded) for sample $i$ and class $c$
- $p_{i,c}$: Predicted probability for sample $i$ and class $c$
- $C$: Number of classes
- $n$: Number of samples

### Formula Breakdown
- $\log(p)$: The logarithm penalizes confident but wrong predictions heavily.
- $y_i$ or $y_{i,c}$: The true label, so only the log probability of the correct class is considered.
- The negative sign ensures that higher probabilities for the correct class reduce the loss.

## Use Cases
- **Logistic Regression**: For binary classification problems.
- **Neural Networks**: For both binary and multi-class classification tasks.
- **Any Classification Problem**: Where the output is a probability distribution over classes.

## Advantages
- **Probabilistic Output**: Encourages the model to output well-calibrated probabilities.
- **Strong Penalization of Confident Errors**: Wrong predictions with high confidence are penalized more.
- **Widely Supported**: Standard in most machine learning libraries.

## Disadvantages
- **Numerical Instability**: If $p_i$ or $p_{i,c}$ is exactly $0$ or $1$, $\log(0)$ is undefined. This is usually handled by clipping the predicted probabilities.
- **Not Suitable for Regression**: Only for classification tasks.

## Python Example
### Binary Cross Entropy
```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
bce = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross Entropy: {bce}")
```

### Categorical Cross Entropy
```python
def categorical_cross_entropy(y_true, y_pred):
    # y_true and y_pred are arrays of shape (n_samples, n_classes)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
cce = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross Entropy: {cce}")
```

### Explanation of the Code
- **Binary Cross Entropy**:
  - `y_true` is a NumPy array of true binary labels: $[1, 0, 1, 1]$
  - `y_pred` is a NumPy array of predicted probabilities: $[0.9, 0.1, 0.8, 0.7]$
  - The function clips predictions to avoid $\log(0)$, then computes the average binary cross entropy loss.
- **Categorical Cross Entropy**:
  - `y_true` is a 2D NumPy array (one-hot encoded) of true labels.
  - `y_pred` is a 2D NumPy array of predicted probabilities for each class.
  - The function computes the sum of $y_{i,c} \log(p_{i,c})$ for each sample, averages over all samples, and returns the loss.
- The output is the cross entropy loss, which quantifies how well the predicted probabilities match the true labels. 