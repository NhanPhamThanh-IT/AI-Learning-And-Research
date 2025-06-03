# Polynomial Kernel

## Introduction
The Polynomial Kernel is a popular kernel function used in Support Vector Machines (SVMs) and other kernelized algorithms. It allows the model to learn non-linear decision boundaries by mapping the input data into a higher-dimensional feature space where the relationship is polynomial. This mapping is performed implicitly through the kernel trick, avoiding the explicit computation of the transformed features.

## Formula
The formula for the Polynomial Kernel between two vectors $ \mathbf{x}_i $ and $ \mathbf{x}_j $ is:

$$ K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d $$

Where:
- $ \mathbf{x}_i $ and $ \mathbf{x}_j $ are the input vectors.
- $ \gamma $ (gamma) is a coefficient that scales the dot product. It affects the influence of a single training example.
- $ r $ (coef0) is an independent term (offset) that can influence the threshold for mapping.
- $ d $ (degree) is the degree of the polynomial. This is an integer parameter.

Using a polynomial kernel of degree $d$ corresponds to mapping the data to a feature space containing all polynomial terms up to degree $d$. For example, with $n$ features and degree $d=2$, the feature space would include terms like $x_k^2$ and $x_k x_l$ for all $k, l$. The dimensionality of this feature space can grow rapidly with the number of features and the degree, but the kernel trick avoids explicit computation in this high-dimensional space.

## Characteristics
- **Non-linearity:** Can model complex non-linear relationships in data.
- **Implicit Mapping:** Leverages the kernel trick to avoid explicit high-dimensional mapping.
- **Hyperparameters:** Requires tuning of $ \gamma $, $ r $, and $ d $. The choice of hyperparameters significantly impacts the model's performance and the complexity of the learned decision boundary.
- **Risk of Overfitting:** For high degrees ($d$), the model can become very flexible and prone to overfitting the training data.
- **Numerical Instability:** Can suffer from numerical instability when the degree $d$ is high due to large feature values in the implicit mapping.

## Applications
The Polynomial Kernel is suitable for problems where a polynomial relationship between features is expected or when a linear kernel is insufficient to capture the data structure. It has been used in various applications, although the RBF kernel is often preferred as a general-purpose non-linear kernel due to having fewer hyperparameters and often performing well.

## Choosing Hyperparameters
Tuning the hyperparameters ($ \gamma $, $ r $, $ d $) is crucial for optimal performance with the polynomial kernel. Cross-validation techniques are typically employed to find the best combination of these parameters on the training data.

- **Degree ($d$):** Higher degrees allow for more complex decision boundaries but increase the risk of overfitting and computational cost. Lower degrees (e.g., 2 or 3) are common starting points.
- **Gamma ($ \gamma $):** Affects the influence of individual training samples. A smaller $ \gamma $ means a larger influence, leading to a smoother decision boundary. A larger $ \gamma $ means a smaller influence, leading to a more complex boundary and potential overfitting. If set to 'auto', it uses $1 / \text{n_features}$. If set to 'scale', it uses $1 / (\text{n_features} \cdot \text{X.var()})$.
- **Coef0 ($r$):** The independent term in the polynomial function. It affects the trade-off between the influence of higher-order versus lower-order polynomial terms.

## Polynomial vs RBF Kernel
While both polynomial and RBF kernels can handle non-linear data, the RBF kernel is often the preferred default choice for several reasons:
- The RBF kernel has fewer hyperparameters to tune (C and gamma) compared to the polynomial kernel (C, degree, gamma, coef0).
- The RBF kernel has been shown to be equivalent to a polynomial kernel of infinite degree under certain conditions, suggesting its power in capturing complex relationships.
- The polynomial kernel can suffer from numerical instability with high degrees or large feature values.
However, the polynomial kernel can be more interpretable if the underlying data generating process is believed to be polynomial. It might also be preferred in specific domains where polynomial features have a clear meaning.

## Example
We can demonstrate the use of the Polynomial Kernel with scikit-learn on a dataset that is not linearly separable. In file `polynomial_kernel_example.py`.

This example illustrates how the Polynomial Kernel allows SVM to find a non-linear decision boundary to separate data that is not linearly separable in the original feature space. 