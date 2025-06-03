# Cosine Similarity and Cosine Distance

## Introduction
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. It is widely used in Machine Learning, particularly in natural language processing and information retrieval, to quantify the similarity between documents or data points represented as vectors.

Unlike Euclidean distance, which measures the magnitude of the difference between vectors, cosine similarity measures the orientation of the vectors. A higher cosine similarity indicates a smaller angle between the vectors, suggesting greater similarity in their direction.

## Formula
For two non-zero vectors $\mathbf{p}$ and $\mathbf{q}$, the cosine similarity is calculated as the dot product of the vectors divided by the product of their magnitudes (L2 norms):

$$ \text{similarity}(\mathbf{p}, \mathbf{q}) = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|} $$

In terms of components, for $n$-dimensional vectors $\mathbf{p} = (p_1, p_2, \dots, p_n)$ and $\mathbf{q} = (q_1, q_2, \dots, q_n)$, the formula is:

$$ \text{similarity}(\mathbf{p}, \mathbf{q}) = \frac{\sum_{i=1}^{n} p_i q_i}{\sqrt{\sum_{i=1}^{n} p_i^2} \sqrt{\sum_{i=1}^{n} q_i^2}} $$

The value of cosine similarity ranges from -1 to 1:
- 1: Indicates that the vectors are identical in direction.
- 0: Indicates that the vectors are orthogonal (perpendicular) and have no similarity.
- -1: Indicates that the vectors are diametrically opposed (pointing in opposite directions).

**Cosine Distance:** Cosine distance is often used as a measure of dissimilarity and is derived from cosine similarity:

$$ \text{distance}(\mathbf{p}, \mathbf{q}) = 1 - \text{similarity}(\mathbf{p}, \mathbf{q}) $$

Cosine distance ranges from 0 to 2.

## Characteristics
- **Measures Orientation:** Cosine similarity is solely focused on the angle between vectors, making it insensitive to their magnitudes. This is useful when the length of the vector is not important (e.g., length of a document in text analysis).
- **Not a Metric:** Cosine distance is not strictly a metric because it does not satisfy the triangle inequality in all cases.
- **Effective for High-Dimensional Sparse Data:** It performs well in high-dimensional spaces, especially with sparse data (data points with many zero values), which is common in text analysis.
- **Requires Non-zero Vectors:** The formula is undefined for zero vectors.

## Comparison with Euclidean Distance
While both measure the relationship between vectors, they do so differently:
- **Euclidean Distance:** Measures the absolute magnitude of the difference. Sensitive to vector magnitude and requires feature scaling for features with different ranges.
- **Cosine Similarity:** Measures the angle between vectors. Insensitive to vector magnitude, making it suitable for data where direction is more important (like text frequency vectors). Less affected by the curse of dimensionality in terms of angle.

## Applications
Cosine similarity and distance are widely used in:
- **Text Analysis:** Measuring similarity between documents, paragraphs, or sentences based on bag-of-words or TF-IDF vector representations.
- **Recommendation Systems:** Finding similar users or items based on their feature vectors.
- **Information Retrieval:** Ranking documents based on their relevance to a search query.
- **Facial Recognition:** Comparing facial feature vectors.
- **Clustering:** Used as a distance measure in some clustering algorithms, particularly for text or high-dimensional data.

## Example
We can illustrate the calculation of cosine similarity using Python. 