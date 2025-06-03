# Ensemble Learning in Machine Learning

## What is Ensemble Learning?
**Ensemble learning** is a powerful machine learning paradigm that combines multiple models (often called "base learners" or "weak learners") to produce a single, superior predictive model. The core idea is that by aggregating the predictions of several models, the ensemble can achieve better generalization, higher accuracy, and greater robustness than any individual model alone.

## Why is Ensemble Learning Important?
- **Improved Accuracy:** Ensembles often outperform single models, especially on complex tasks.
- **Reduced Overfitting:** By combining diverse models, ensembles can reduce the risk of overfitting to the training data.
- **Increased Robustness:** Ensembles are less sensitive to the peculiarities of a single model or dataset split.
- **Flexibility:** Different ensemble methods can be tailored to various tasks and data characteristics.

## Types of Ensemble Learning Methods
Ensemble methods can be broadly categorized into four main types, each with its own strategy for combining models:

### 1. Bagging (Bootstrap Aggregating)
- **How it works:** Trains multiple instances of the same base model on different random subsets (with replacement) of the training data. Predictions are aggregated by averaging (regression) or majority voting (classification).
- **Best for:** Reducing variance and overfitting, especially for high-variance models like decision trees.
- **Example:** Random Forest.
- **[See details and Python example.](01_Bagging/README.md)**

### 2. Boosting
- **How it works:** Trains models sequentially, where each new model focuses on correcting the errors of the previous ones. Final predictions are a weighted combination of all models.
- **Best for:** Reducing both bias and variance, improving accuracy on difficult tasks.
- **Popular algorithms:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost.
- **[See details and Python example.](02_Boosting/README.md)**

### 3. Voting
- **How it works:** Combines predictions from multiple different models (can be of different types). Uses majority voting (hard voting) or average predicted probabilities (soft voting) to make the final prediction.
- **Best for:** Leveraging the strengths of diverse models, simple implementation.
- **[See details and Python example.](03_Voting/README.md)**

### 4. Stacking (Stacked Generalization)
- **How it works:** Trains several base models and then uses a meta-model to learn how to best combine their predictions. The meta-model is trained on the outputs of the base models.
- **Best for:** Capturing complex relationships between model predictions, achieving top performance in competitions.
- **[See details and Python example.](04_Stacking/README.md)**

## How to Choose an Ensemble Method
- **Bagging:** Use when you want to reduce variance and overfitting, especially with unstable models (e.g., decision trees).
- **Boosting:** Use when you need to reduce both bias and variance, and want to focus on hard-to-predict cases.
- **Voting:** Use when you have several strong but different models and want a simple way to combine them.
- **Stacking:** Use when you want to maximize predictive performance and are comfortable with more complex model architectures.

## Further Reading and Examples
Each subfolder contains a detailed explanation and a Python example for the respective ensemble method. Explore them for theoretical background, use cases, advantages/disadvantages, and code implementations:
- [01_Bagging](01_Bagging/README.md)
- [02_Boosting](02_Boosting/README.md)
- [03_Voting](03_Voting/README.md)
- [04_Stacking](04_Stacking/README.md)

---

**References:**
- Breiman, L. (1996). Bagging predictors. Machine Learning.
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences.
- Kuncheva, L. I. (2004). Combining Pattern Classifiers: Methods and Algorithms. Wiley.
- Wolpert, D. H. (1992). Stacked generalization. Neural Networks.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly.
