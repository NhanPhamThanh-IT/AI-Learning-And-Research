# Radial Basis Function (RBF) Kernel

## Introduction
The **Radial Basis Function (RBF) kernel**, also known as the **Gaussian kernel**, is one of the most widely used and powerful kernel functions in Support Vector Machines (SVMs) and other kernelized methods. It is a general-purpose kernel that can handle non-linear relationships between data points and is often a good default choice.

The RBF kernel implicitly maps data into a potentially infinite-dimensional feature space. In this high-dimensional space, it measures the similarity between two points based on their distance from a central point (or a support vector in the context of SVM). The similarity decreases as the distance from the center increases, following a Gaussian (bell-shaped) function. This allows the SVM to create highly flexible, non-linear decision boundaries that can separate complex datasets.

## Formula
The formula for the RBF Kernel between two vectors $ \mathbf{x}_i $ and $ \mathbf{x}_j $ is:

$$ K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $$

Where:
- $ \mathbf{x}_i $ and $ \mathbf{x}_j $ are the input vectors.
- $ \|\mathbf{x}_i - \mathbf{x}_j\|^2 $ is the squared Euclidean distance between the two vectors.
- $ \gamma $ (gamma) is a hyperparameter that defines how much influence a single training example has. It can be seen as the inverse of the radius of influence of samples selected by the model as support vectors. A small $ \gamma $ means a large influence (points far away still affect the boundary, leading to a smoother decision boundary), while a large $ \gamma $ means a small influence (only points very close affect the boundary, leading to a more complex, potentially jagged boundary and potential overfitting). $ \gamma $ must be greater than 0. In scikit-learn, $ \gamma $ can be set to 'scale' (default) which uses $1 / (\text{n_features} \cdot \text{X.var()})$ or 'auto' which uses $1 / \text{n_features}$.
- $ \exp $ is the exponential function.

The RBF kernel measures the similarity of $ \mathbf{x}_i $ and $ \mathbf{x}_j $ based on their proximity in the original input space. The closer the points are, the higher their kernel value (closer to 1). As they move farther apart, the kernel value decreases towards 0. This local nature of the RBF kernel allows it to capture complex local patterns in the data.

## Characteristics
- **Powerful Non-linearity:** Capable of modeling highly complex and arbitrary non-linear decision boundaries.
- **Infinite-Dimensional Space:** Implicitly maps data into a potentially infinite-dimensional feature space, allowing for complex separations without the computational cost of explicit mapping.
- **General Purpose:** Often performs well on a wide variety of datasets and is a common go-to kernel when linearity is insufficient.
- **Hyperparameters:** Has one primary kernel-specific hyperparameter, $ \gamma $. The SVM regularization parameter $ C $ is also crucial and interacts with $ \gamma $.
- **Sensitivity to Feature Scaling:** Since the kernel relies on Euclidean distance, features with larger scales will have a disproportionately larger impact. Therefore, scaling your features (e.g., using `StandardScaler` or `MinMaxScaler`) is highly recommended before using the RBF kernel.
- **Computational Cost:** The training and prediction time for SVMs with RBF kernels can be slower than with linear kernels, especially for very large datasets, as the complexity can depend on the number of support vectors.

## Applications
The RBF kernel is widely applied in various domains where non-linear relationships are prevalent:
- **Image Classification:** Used for classifying images based on various feature representations.
- **Handwriting Recognition:** Effective in classifying handwritten digits or characters.
- **Bioinformatics:** Applications in protein structure prediction, gene expression analysis, and sequence alignment.
- **Financial Forecasting:** Used for regression and classification tasks on financial data.
- **Any task with Non-linear Data:** It's a strong candidate whenever your data is not linearly separable and you need a flexible model.

It is a solid choice when you don't have prior knowledge about the data structure and a linear kernel doesn't perform well.

## Choosing Hyperparameters ($ \gamma $ and $ C $)
The performance of an SVM with an RBF kernel is highly dependent on the hyperparameters $ \gamma $ and the regularization parameter $ C $. Tuning these parameters is critical and typically involves searching over a range of values using cross-validation.

- **$ C $ (Regularization):** This parameter controls the trade-off between achieving a low training error and a large margin. 
    - **Small $ C $:** Penalizes misclassifications less. Leads to a larger margin and potentially more misclassifications on the training data (underfitting). More regularization.
    - **Large $ C $:** Penalizes misclassifications heavily. Leads to a smaller margin and aims to classify all training examples correctly (potential overfitting). Less regularization.
- **$ \gamma $ (Kernel Coefficient):** This parameter defines the influence of a single training example. 
    - **Small $ \gamma $:** A large radius of influence. Leads to a smoother decision boundary. The model considers points far away. Can lead to underfitting if too small.
    - **Large $ \gamma $:** A small radius of influence. Leads to a highly complex, potentially jagged decision boundary. Only points very close are considered. Can lead to overfitting if too large, as the model becomes too sensitive to individual training points.

A common strategy for tuning is to perform a grid search over exponentially increasing values of $ C $ and $ \gamma $ (e.g., $ C \in \{0.1, 1, 10, 100\} $, $ \gamma \in \{0.001, 0.01, 0.1, 1\} $). \textit{Note: Always scale your data before tuning $ \gamma $}.

## RBF vs Polynomial Kernel
As mentioned in the main README, the RBF kernel is often preferred over the polynomial kernel as a default non-linear kernel due to:
- **Fewer Hyperparameters:** RBF has two main hyperparameters ($C$, $ \gamma $), while polynomial has four ($C$, $d$, $ \gamma $, $r$). This simplifies the tuning process.
- **General Performance:** The RBF kernel is a universal approximator, meaning it can approximate any continuous function. While the polynomial kernel of infinite degree is also a universal approximator, the RBF often performs well in practice without needing to select a specific degree.
- **Numerical Stability:** RBF is generally less prone to numerical issues than high-degree polynomial kernels.

However, if there is strong prior knowledge suggesting a polynomial relationship in the data, the polynomial kernel might be a better fit.

## Example
We can demonstrate the use of the RBF Kernel with scikit-learn on a non-linearly separable dataset and compare its decision boundary with a linear kernel. In file `rbf_kernel_example.py`.

This example highlights the RBF kernel's ability to create complex, non-linear decision boundaries, making it suitable for datasets where a linear separation is not possible. 