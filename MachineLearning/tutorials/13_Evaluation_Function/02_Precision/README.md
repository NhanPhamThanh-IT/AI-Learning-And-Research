# Precision Metric

## Definition
Precision is an evaluation metric for classification models, especially useful in cases where the cost of false positives is high. It measures the proportion of positive predictions that are actually correct.

## Mathematical Formula
For binary classification:

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

Where:
- $TP$: True Positives (correctly predicted positive cases)
- $FP$: False Positives (incorrectly predicted positive cases)

### Formula Breakdown
- The numerator $TP$ is the number of correct positive predictions.
- The denominator $(TP + FP)$ is the total number of predicted positive cases.

## Use Cases
- **Information Retrieval**: When it is important that returned results are relevant (e.g., search engines).
- **Medical Diagnosis**: When false positives can lead to unnecessary treatments.

## Advantages
- **Focuses on Positive Class**: Useful when the positive class is more important.
- **Reduces False Positives**: High precision means few false positives.

## Disadvantages
- **Ignores False Negatives**: Does not consider cases where positive samples are missed.
- **May Not Reflect Overall Performance**: Should be used with other metrics for a complete picture.

## Python Example
```python
import numpy as np

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
prec = precision_score(y_true, y_pred)
print(f"Precision: {prec}")
```

### Explanation of the Code
- `y_true` is a NumPy array of true class labels.
- `y_pred` is a NumPy array of predicted class labels.
- The function `precision_score` counts true positives and false positives, then computes precision using the formula above.
- The output is the precision, representing the proportion of positive predictions that are correct. 