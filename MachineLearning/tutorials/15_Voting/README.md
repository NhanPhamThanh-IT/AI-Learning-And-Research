# Voting Ensemble

## Introduction
Voting is an ensemble learning technique that combines the predictions from multiple different models to improve overall performance. It is commonly used for classification tasks and can be applied to both homogeneous and heterogeneous ensembles.

## Theoretical Background
In a voting ensemble, several base models are trained independently. Their predictions are combined using either majority voting (for classification) or averaging (for regression). There are two main types:
- **Hard Voting**: The class with the most votes is chosen.
- **Soft Voting**: The class with the highest average predicted probability is chosen.

## Example
A voting classifier might combine a logistic regression, a decision tree, and a support vector machine. Each model votes on the class label, and the majority wins (hard voting) or the probabilities are averaged (soft voting).

## Applications
- Classification tasks where different models have complementary strengths
- Competitions (e.g., Kaggle) where ensemble methods often win

## Advantages
- Simple to implement
- Can leverage strengths of different models
- Reduces risk of choosing a poorly performing single model

## Disadvantages
- May not perform better if base models are highly correlated
- Less interpretable than single models
- Requires all models to be trained and maintained

## Comparison with Other Ensemble Methods
- **Bagging**: Uses the same base model type, trained on different data samples.
- **Boosting**: Sequentially trains models, each focusing on previous errors.
- **Stacking**: Trains a meta-model to combine base model predictions.

## References
- Kuncheva, L. I. (2004). Combining Pattern Classifiers: Methods and Algorithms. Wiley.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 