# K-Nearest Neighbors (KNN)

## Introduction
K-Nearest Neighbors (KNN) is a simple, non-parametric, and instance-based learning algorithm used for classification and regression. It predicts the label of a data point based on the labels of its nearest neighbors in the feature space. KNN is known for its simplicity and effectiveness in many practical applications.

## Theoretical Background
Given a query point, KNN finds the \( k \) closest points in the training data (using a distance metric such as Euclidean distance) and assigns the most common class (for classification) or the average value (for regression).

### Distance Metrics
- **Euclidean Distance**: Most common for continuous variables
- **Manhattan Distance**: For grid-like data
- **Minkowski Distance**: Generalization of Euclidean and Manhattan
- **Cosine Similarity**: Measures the cosine of the angle between two vectors (useful for text data)

### Choosing k
- Small \( k \): Sensitive to noise, may overfit
- Large \( k \): Smoother decision boundary, may underfit
- Odd \( k \): Helps avoid ties in binary classification

## Algorithm Steps
1. Choose the number of neighbors \( k \)
2. Calculate the distance between the query point and all training points
3. Select the \( k \) nearest neighbors
4. For classification: assign the class most common among the neighbors
5. For regression: compute the average of the neighbors' values

## Example
Suppose we want to classify a new flower based on its petal and sepal measurements. KNN will find the k most similar flowers in the training set and assign the most common species among them to the new flower.

## Advantages
- Simple to implement and understand
- No training phase (lazy learning)
- Naturally handles multi-class problems
- Can adapt to complex decision boundaries

## Disadvantages
- Computationally expensive for large datasets (slow prediction)
- Sensitive to irrelevant features and the scale of data
- Performance depends on the choice of \( k \) and distance metric
- Does not work well with high-dimensional data (curse of dimensionality)

## Applications
- Recommendation systems
- Image recognition
- Anomaly detection
- Text categorization
- Medical diagnosis

## Comparison with Other Methods
- **Decision Trees**: Faster prediction, but may overfit
- **Logistic Regression**: Assumes linear boundaries, less flexible
- **Support Vector Machines**: More robust to high-dimensional data, but more complex

## Practical Considerations
- Feature scaling (normalization/standardization) is important
- Use efficient data structures (KD-Tree, Ball Tree) for large datasets

## References
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory.
- Duda, R. O., Hart, P. E., & Stork, D. G. (2001). Pattern Classification. Wiley.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 