# Decision Tree

## Introduction
A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It models decisions and their possible consequences as a tree-like structure of nodes. Decision trees are popular due to their interpretability and ability to handle both numerical and categorical data.

## Theoretical Background
A decision tree consists of:
- **Root Node**: The top node representing the entire dataset
- **Internal Nodes**: Represent tests on features
- **Branches**: Outcomes of the tests
- **Leaf Nodes**: Final output or class label

The tree is built by recursively splitting the dataset based on feature values that maximize a certain criterion (e.g., information gain, Gini impurity).

### Splitting Criteria
- **Information Gain (Entropy)**: Measures the reduction in entropy after a dataset is split on an attribute. Used in ID3 and C4.5 algorithms.
- **Gini Impurity**: Measures the frequency at which a randomly chosen element would be incorrectly labeled. Used in CART algorithm.
- **Chi-Square**: Measures statistical significance of splits.

### Stopping Criteria
- All samples in a node belong to the same class
- Maximum tree depth is reached
- Minimum number of samples per node

## Algorithm Steps
1. Select the best feature to split the data (using a criterion)
2. Split the dataset into subsets
3. Repeat recursively for each subset
4. Stop when a stopping criterion is met

## Example
Suppose we want to classify whether a person will play tennis based on weather conditions (Outlook, Temperature, Humidity, Wind). The decision tree will split the data based on the most informative features, creating a path from root to leaf for each possible outcome.

## Advantages
- Easy to interpret and visualize
- Handles both numerical and categorical data
- Requires little data preprocessing
- Can capture non-linear relationships

## Disadvantages
- Prone to overfitting, especially with deep trees
- Unstable to small variations in data
- Can create biased trees if some classes dominate
- Greedy splitting may not yield globally optimal trees

## Applications
- Credit scoring
- Medical diagnosis
- Customer segmentation
- Fraud detection

## Comparison with Other Methods
- **Random Forests**: Ensemble of decision trees, reduces overfitting
- **Logistic Regression**: Assumes linear relationships, less flexible
- **Neural Networks**: Can model complex patterns but less interpretable

## References
- Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning.
- Breiman, L. et al. (1984). Classification and Regression Trees. Wadsworth.
- Mitchell, T. M. (1997). Machine Learning. McGraw-Hill. 