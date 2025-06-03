# Euclidean Distance

## Introduction
Euclidean distance is the most common and intuitively understandable measure of the straight-line distance between two points in Euclidean space. It is a fundamental concept in geometry, named after the ancient Greek mathematician Euclid of Alexandria. It is widely used as a distance metric in various fields, including Machine Learning.

In the context of Machine Learning, data points are often represented as vectors in a multi-dimensional feature space. Euclidean distance is used to quantify the dissimilarity between two such data points.

## Formula
For two points $p$ and $q$ in an $n$-dimensional Euclidean space, where $p = (p_1, p_2, \dots, p_n)$ and $q = (q_1, q_2, \dots, q_n)$, the Euclidean distance $d(p, q)$ is calculated using the Pythagorean theorem:

$$ d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2} $$

This can be written in summation notation as:

$$ d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} $$

Where:
- $p_i$ and $q_i$ are the $i$-th coordinates of points $p$ and $q$, respectively.
- $n$ is the number of dimensions.

**Squared Euclidean Distance:** Sometimes, the squared Euclidean distance is used, which omits the square root. This saves computational cost and is sufficient when only comparing distances (since the square function is monotonic for non-negative values). The formula for squared Euclidean distance is:

$$ d^2(p, q) = \sum_{i=1}^{n} (p_i - q_i)^2 $$

## Characteristics
- **Metric:** Euclidean distance is a true metric, satisfying non-negativity, identity of indiscernibles, symmetry, and triangle inequality.
- **Sensitivity to Magnitude:** Euclidean distance is sensitive to the magnitude of the differences between coordinates. Features with larger scales can dominate the distance calculation.
- **Dimensionality:** As the number of dimensions increases, the concept of Euclidean distance can become less intuitive, and distances between points tend to become very similar (curse of dimensionality).
- **Requires Feature Scaling:** Due to its sensitivity to magnitude, it's often necessary to scale features (e.g., using standardization or normalization) before using Euclidean distance, especially if features have different units or ranges.

## Applications
Euclidean distance is widely used in various Machine Learning algorithms:
- **K-Means Clustering:** Used to assign data points to the nearest cluster centroid.
- **K-Nearest Neighbors (KNN):** Used to find the $k$ closest training examples to a new data point for classification or regression.
- **Principal Component Analysis (PCA):** Euclidean distance is preserved after PCA transformation.
- ** dimensionality Reduction (e.g., MDS):** Used in some dimensionality reduction techniques.
- **Anomaly Detection:** Measuring the distance of a point from a cluster or known examples.

## Example
We can illustrate the calculation of Euclidean distance using Python. 