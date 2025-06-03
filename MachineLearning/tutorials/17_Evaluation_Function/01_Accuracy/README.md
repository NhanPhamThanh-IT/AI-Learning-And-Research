# Accuracy Metric

## Definition
Accuracy is one of the most basic and widely used evaluation metrics for classification problems. It measures the proportion of correct predictions among the total number of cases examined.

## Mathematical Formula
For $n$ samples:

$$
\mathrm{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

Or, in terms of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN):

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:
- $TP$: True Positives (correctly predicted positive cases)
- $TN$: True Negatives (correctly predicted negative cases)
- $FP$: False Positives (incorrectly predicted positive cases)
- $FN$: False Negatives (incorrectly predicted negative cases)

### Formula Breakdown
- The numerator $(TP + TN)$ is the total number of correct predictions.
- The denominator $(TP + TN + FP + FN)$ is the total number of samples.

## Use Cases
- **Balanced Classification Problems**: When the classes are roughly equally distributed.
- **Quick Model Evaluation**: Provides a simple, intuitive measure of performance.

## Advantages
- **Simple to Interpret**: Easy to understand and calculate.
- **Widely Used**: Standard metric for many classification tasks.

## Disadvantages
- **Misleading for Imbalanced Data**: High accuracy can be achieved by always predicting the majority class.
- **Does Not Distinguish Types of Errors**: Treats all errors equally, regardless of their type.

## Python Example
```python
import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc}")
```

### Explanation of the Code
- `y_true` is a NumPy array of true class labels.
- `y_pred` is a NumPy array of predicted class labels.
- The function `accuracy_score` compares each prediction to the true label, counts the number of correct predictions, and divides by the total number of samples.
- The output is the accuracy, representing the proportion of correct predictions. 