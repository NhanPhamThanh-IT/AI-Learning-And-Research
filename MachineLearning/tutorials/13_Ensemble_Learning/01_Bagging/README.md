# Bagging (Bootstrap Aggregating)

## Introduction
Bagging, short for Bootstrap Aggregating, is an ensemble learning technique designed to improve the stability and accuracy of machine learning algorithms. It reduces variance and helps to avoid overfitting, especially for high-variance models like decision trees.

## Theoretical Background
Bagging works by generating multiple versions of a predictor and using these to get an aggregated predictor. Each model is trained on a random bootstrap sample (sampling with replacement) of the original dataset. The final prediction is made by averaging (for regression) or majority voting (for classification) the predictions of all models.

### Steps of Bagging:
1. Draw multiple bootstrap samples from the training data.
2. Train a base model (e.g., decision tree) on each sample.
3. Aggregate the predictions from all models (average or vote).

## Example
A common example of bagging is the Random Forest algorithm, which builds an ensemble of decision trees using bagging and random feature selection.

## Applications
- Classification and regression tasks
- Reducing overfitting in high-variance models
- Used in Random Forests, bagged SVMs, etc.

## Advantages
- Reduces variance and overfitting
- Simple to implement
- Can improve accuracy for unstable models

## Disadvantages
- Less effective for low-variance models
- Increased computational cost
- Less interpretable than single models

## Comparison with Other Ensemble Methods
- **Boosting**: Focuses on reducing bias by sequentially training models, each correcting the errors of the previous.
- **Voting**: Combines predictions from different model types, not just the same base learner.
- **Stacking**: Learns how to best combine predictions using a meta-model.

## References
- Breiman, L. (1996). Bagging predictors. Machine Learning.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 