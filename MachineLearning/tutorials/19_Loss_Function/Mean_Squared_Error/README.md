# Mean Squared Error (MSE) Loss Function

## Definition
Mean Squared Error (MSE) is a fundamental loss function used primarily in regression tasks. It quantifies the average of the squares of the differences between the actual values and the predicted values. The lower the MSE, the closer the predicted values are to the actual values.

## Mathematical Formula
The MSE for $n$ samples is defined as:

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$: The actual (true) value for the $i$-th sample
- $\hat{y}_i$: The predicted value for the $i$-th sample
- $n$: The total number of samples

### Formula Breakdown
- The difference $(y_i - \hat{y}_i)$ measures the error for each sample.
- Squaring the error $(y_i - \hat{y}_i)^2$ ensures all errors are positive and penalizes larger errors more heavily.
- Summing over all samples $\sum_{i=1}^{n}$ gives the total squared error.
- Dividing by $n$ gives the average squared error per sample.

## Use Cases
- **Linear Regression**: MSE is the default loss function for fitting linear models.
- **Polynomial Regression**: Used to fit higher-order curves to data.
- **General Regression Problems**: Any scenario where the goal is to minimize the average squared difference between predictions and actual values.

## Advantages
- **Penalizes Large Errors**: Squaring the errors means larger mistakes are penalized more, which can be useful if large errors are particularly undesirable.
- **Differentiable**: The function is smooth and differentiable, making it suitable for gradient-based optimization algorithms.
- **Simple Interpretation**: Represents the average squared error per sample.

## Disadvantages
- **Sensitive to Outliers**: Because errors are squared, outliers (large errors) have a disproportionately large effect on the MSE.
- **Not Robust**: In datasets with many outliers, MSE may not represent the typical error well.

## Python Example
```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")
```

### Explanation of the Code
- `y_true` is a NumPy array containing the actual values: $[3.0, -0.5, 2.0, 7.0]$
- `y_pred` is a NumPy array containing the predicted values: $[2.5, 0.0, 2.0, 8.0]$
- The function `mean_squared_error` computes the squared differences $(y_i - \hat{y}_i)^2$ for each pair, takes the mean (average), and returns the result.
- The output is the MSE for these predictions, which quantifies the average squared error between the predictions and the actual values. 