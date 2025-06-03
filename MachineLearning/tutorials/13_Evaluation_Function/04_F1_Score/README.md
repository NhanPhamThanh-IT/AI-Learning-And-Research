# F1 Score Metric

## Definition
The F1 Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both the precision and the recall of a classification model, especially useful when you need a balance between the two.

## Mathematical Formula
For binary classification:

$$
\mathrm{F1\ Score} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

Where:
- $\mathrm{Precision}$: The precision of the model
- $\mathrm{Recall}$: The recall of the model

### Formula Breakdown
- The numerator $2 \times (\mathrm{Precision} \times \mathrm{Recall})$ emphasizes the importance of both metrics.
- The denominator $(\mathrm{Precision} + \mathrm{Recall})$ ensures the score is high only when both precision and recall are high.
- The F1 Score ranges from $0$ (worst) to $1$ (best).

## Use Cases
- **Imbalanced Datasets**: When you need a single metric that balances precision and recall.
- **Model Selection**: Useful for comparing models when both false positives and false negatives are important.

## Advantages
- **Balances Precision and Recall**: Useful when you need to consider both types of errors.
- **Robust to Class Imbalance**: More informative than accuracy on imbalanced datasets.

## Disadvantages
- **Ignores True Negatives**: Does not take true negatives into account.
- **May Not Reflect Overall Performance**: Should be used with other metrics for a complete picture.

## Python Example
```python
import numpy as np

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")
```

### Explanation of the Code
- `precision_score` and `recall_score` are helper functions to compute precision and recall.
- `f1_score` computes the F1 Score using the formula above.
- `y_true` and `y_pred` are NumPy arrays of true and predicted class labels.
- The output is the F1 Score, which balances precision and recall in a single metric. 