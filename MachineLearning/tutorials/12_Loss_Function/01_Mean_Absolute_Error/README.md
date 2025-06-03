# Mean Absolute Error (MAE) Loss Function

## Definition
Mean Absolute Error (MAE) is a widely used loss function for regression tasks. It measures the average magnitude of the errors between predicted and actual values, without considering their direction. MAE gives an idea of how wrong the predictions are, on average.

## Mathematical Formula
The MAE for $n$ samples is defined as:

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- $y_i$: The actual (true) value for the $i$-th sample
- $\hat{y}_i$: The predicted value for the $i$-th sample
- $n$: The total number of samples

### Formula Breakdown
- The absolute error $|y_i - \hat{y}_i|$ measures the magnitude of the difference for each sample, ignoring whether the prediction is above or below the actual value.
- Summing over all samples $\sum_{i=1}^{n}$ gives the total absolute error.
- Dividing by $n$ gives the average absolute error per sample.

## Use Cases
- **Robust Regression**: MAE is preferred when you want a loss function that is less sensitive to outliers.
- **Fair Error Weighting**: When all errors should be weighted equally, regardless of their size.
- **Model Evaluation**: Used as a metric to evaluate regression models.

## Advantages
- **Robust to Outliers**: MAE does not heavily penalize large errors, making it more robust to outliers than MSE.
- **Easy to Interpret**: The result is in the same unit as the original data, making it easy to understand.

## Disadvantages
- **Not Differentiable at Zero**: The absolute value function is not differentiable at zero, which can make optimization more challenging for some algorithms.
- **May Converge Slower**: In some cases, optimization using MAE may converge slower than with MSE.

## Python Example
```python
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Example usage
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae}")
```

### Explanation of the Code
- `y_true` is a NumPy array containing the actual values: $[3.0, -0.5, 2.0, 7.0]$
- `y_pred` is a NumPy array containing the predicted values: $[2.5, 0.0, 2.0, 8.0]$
- The function `mean_absolute_error` computes the absolute differences $|y_i - \hat{y}_i|$ for each pair, takes the mean (average), and returns the result.
- The output is the MAE for these predictions, which quantifies the average magnitude of the errors between the predictions and the actual values. 