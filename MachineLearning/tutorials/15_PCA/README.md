# Principal Component Analysis (PCA)

## Introduction
Principal Component Analysis (PCA) is an unsupervised linear dimensionality reduction technique. It transforms the original features into a new set of orthogonal features (principal components) that capture the maximum variance in the data.

## Theoretical Background
PCA projects the data onto a lower-dimensional space by finding the directions (principal components) that maximize the variance. The first principal component captures the most variance, the second is orthogonal to the first and captures the next most, and so on.

### The PCA Algorithm
1. Standardize the data (zero mean, unit variance).
2. Compute the covariance matrix of the data.
3. Calculate the eigenvectors and eigenvalues of the covariance matrix.
4. Sort the eigenvectors by decreasing eigenvalues (variance explained).
5. Select the top k eigenvectors to form the principal components.
6. Project the data onto the new subspace.

### Mathematical Formulation
Given a data matrix \(X\), PCA solves:
$$
\text{maximize}_w \quad \text{Var}(Xw) \quad \text{subject to} \quad \|w\|=1
$$

## Example
Suppose we have a dataset with 3 features. PCA can reduce it to 2 principal components that capture most of the variance, allowing for visualization and noise reduction.

## Applications
- Data visualization
- Noise reduction
- Feature extraction
- Preprocessing for supervised learning
- Image compression

## Advantages
- Reduces dimensionality and noise
- Improves computational efficiency
- Decorrelates features
- Useful for visualization

## Disadvantages
- Linear method: cannot capture non-linear relationships
- Principal components may be hard to interpret
- Sensitive to scaling
- May discard useful information if too many components are dropped

## Comparison with Other Dimensionality Reduction Methods
- **t-SNE, UMAP**: Non-linear, better for visualization but less interpretable
- **LDA**: Supervised, maximizes class separability
- **Autoencoders**: Non-linear, neural network-based

## References
- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 