# K-Means Clustering

## Introduction
K-Means Clustering is an unsupervised machine learning algorithm used to partition a dataset into K distinct, non-overlapping clusters. It is one of the most popular clustering algorithms due to its simplicity and efficiency.

## Theoretical Background
K-Means aims to group data points such that points in the same cluster are more similar to each other than to those in other clusters. The similarity is typically measured using Euclidean distance.

### The K-Means Algorithm
1. Choose the number of clusters K.
2. Initialize K centroids randomly.
3. Assign each data point to the nearest centroid (cluster assignment step).
4. Update the centroids by calculating the mean of all points assigned to each cluster (centroid update step).
5. Repeat steps 3 and 4 until convergence (no change in assignments or centroids).

### Objective Function
K-Means minimizes the within-cluster sum of squares (WCSS):
\[
J = \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2
\]
where \(C_i\) is the set of points in cluster i and \(\mu_i\) is the centroid of cluster i.

## Example
Suppose we have a dataset of points in 2D space. K-Means can be used to group them into K clusters, each represented by a centroid. The algorithm iteratively refines the centroids and assignments until the clusters are stable.

## Applications
- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection
- Market basket analysis

## Advantages
- Simple and fast
- Scales well to large datasets
- Works well when clusters are spherical and equally sized

## Disadvantages
- Requires specifying K in advance
- Sensitive to initial centroid placement
- Struggles with non-spherical or overlapping clusters
- Sensitive to outliers

## Comparison with Other Clustering Methods
- **Hierarchical Clustering**: Does not require specifying K, can capture nested clusters
- **DBSCAN**: Can find arbitrarily shaped clusters, robust to outliers
- **Gaussian Mixture Models**: Probabilistic, can model elliptical clusters

## References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 