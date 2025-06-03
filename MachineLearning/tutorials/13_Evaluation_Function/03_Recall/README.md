# Recall Metric

## Definition
Recall (also known as Sensitivity or True Positive Rate) is an evaluation metric for classification models. It measures the proportion of actual positive cases that are correctly identified by the model.

## Mathematical Formula
For binary classification:

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

Where:
- $TP$: True Positives (correctly predicted positive cases)
- $FN$: False Negatives (actual positive cases missed by the model)

### Formula Breakdown
- The numerator $TP$ is the number of correct positive predictions.
- The denominator $(TP + FN)$ is the total number of actual positive cases.

## Use Cases
- **Medical Screening**: When it is important to identify as many positive cases as possible (e.g., disease detection).
- **Fraud Detection**: When missing a positive case is costly.

## Advantages
- **Focuses on Positive Cases**: Useful when missing positive cases is more costly than false alarms.
- **Complements Precision**: Helps provide a fuller picture of model performance.

## Disadvantages
- **Ignores False Positives**: Does not consider cases where negative samples are incorrectly labeled as positive.
- **May Not Reflect Overall Performance**: Should be used with other metrics for a complete picture.

## Python Example
```python
import numpy as np

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
rec = recall_score(y_true, y_pred)
print(f"Recall: {rec}")
```

### Explanation of the Code
- `y_true` is a NumPy array of true class labels.
- `y_pred` is a NumPy array of predicted class labels.
- The function `recall_score` counts true positives and false negatives, then computes recall using the formula above.
- The output is the recall, representing the proportion of actual positive cases that are correctly identified. 