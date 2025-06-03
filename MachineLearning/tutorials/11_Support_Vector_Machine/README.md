# Support Vector Machine (SVM)

## Introduction
Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. It aims to find the optimal hyperplane that best separates data points of different classes. SVMs are known for their effectiveness in high-dimensional spaces and their ability to model complex decision boundaries using kernel functions.

## Theoretical Background
Given a set of labeled training data, SVM finds the hyperplane that maximizes the margin between classes. The data points closest to the hyperplane are called support vectors. The margin is the distance between the hyperplane and the nearest data points from each class.

For linearly separable data, the decision boundary is defined as:
\[
\mathbf{w}^T \mathbf{x} + b = 0
\]
where \( \mathbf{w} \) is the weight vector and \( b \) is the bias.

### Objective Function
The SVM optimization problem is:
\[
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
\]
subject to:
\[
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1
\]
for all training samples \( (\mathbf{x}_i, y_i) \).

### Kernel Trick
For non-linearly separable data, SVM uses kernel functions to map data into higher-dimensional space where a linear separator can be found. Common kernels include:
- **Linear Kernel**: \( K(x, x') = x^T x' \)
- **Polynomial Kernel**: \( K(x, x') = (\gamma x^T x' + r)^d \)
- **Radial Basis Function (RBF) Kernel**: \( K(x, x') = \exp(-\gamma \|x - x'\|^2) \)
- **Sigmoid Kernel**: \( K(x, x') = \tanh(\gamma x^T x' + r) \)

## Example
Suppose we want to classify emails as spam or not spam. SVM can find the optimal boundary in a high-dimensional feature space (e.g., word frequencies) to separate the two classes, even if the data is not linearly separable.

## Advantages
- Effective in high-dimensional spaces
- Works well with clear margin of separation
- Versatile with different kernel functions
- Robust to overfitting (especially with proper regularization)

## Disadvantages
- Not suitable for very large datasets (high computational cost)
- Less effective when classes overlap significantly
- Requires careful parameter and kernel selection
- Harder to interpret than decision trees

## Applications
- Text classification (spam detection, sentiment analysis)
- Image recognition
- Bioinformatics (gene classification)
- Handwriting recognition

## Comparison with Other Methods
- **Logistic Regression**: Simpler, but less powerful for non-linear problems
- **Decision Trees/Random Forests**: More interpretable, can handle missing values
- **Neural Networks**: Can model more complex patterns, but require more data and tuning

## References
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Scholkopf, B., & Smola, A. J. (2001). Learning with Kernels. MIT Press. 