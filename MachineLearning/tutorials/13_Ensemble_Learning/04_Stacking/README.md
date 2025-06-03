# Stacking Ensemble

## Introduction
Stacking (Stacked Generalization) is an advanced ensemble learning technique that combines multiple base models using a meta-model (or blender). The meta-model learns how to best combine the predictions of the base models to improve overall performance.

## Theoretical Background
In stacking, several base models (level-0 models) are trained on the original dataset. Their predictions are then used as input features for a meta-model (level-1 model), which is trained to make the final prediction. This allows the ensemble to learn how to correct the weaknesses of individual models.

### Steps of Stacking:
1. Train several base models on the training data.
2. Use the base models to generate predictions (out-of-fold or on a validation set).
3. Train a meta-model on these predictions.
4. For new data, base models make predictions, which are then combined by the meta-model.

## Example
A stacking ensemble might use a random forest, a support vector machine, and a logistic regression as base models, with a gradient boosting machine as the meta-model.

## Applications
- Classification and regression tasks
- Competitions (e.g., Kaggle) where stacking is often used for top solutions
- Problems where different models capture different aspects of the data

## Advantages
- Can capture complex relationships between base model predictions
- Often achieves higher accuracy than bagging, boosting, or voting
- Flexible: can use any combination of models

## Disadvantages
- More complex to implement and tune
- Risk of overfitting if not properly validated
- Requires careful data splitting to avoid information leakage

## Comparison with Other Ensemble Methods
- **Bagging**: Aggregates predictions from the same model type, trained on different samples.
- **Boosting**: Sequentially trains models to focus on previous errors.
- **Voting**: Simple combination of model predictions, no meta-model.

## References
- Wolpert, D. H. (1992). Stacked generalization. Neural Networks.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. 