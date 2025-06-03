# Evaluation Metrics in Machine Learning

## What is an Evaluation Metric?
An **evaluation metric** is a quantitative measure used to assess how well a machine learning model performs on a given dataset. Evaluation metrics provide a way to objectively compare models, guide model selection, and understand the strengths and weaknesses of different approaches.

## Why are Evaluation Metrics Important?
Evaluation metrics are essential for:
- **Model Selection:** Choosing the best model among alternatives.
- **Performance Monitoring:** Tracking model performance over time or across datasets.
- **Understanding Trade-offs:** Different metrics highlight different aspects of model performance (e.g., accuracy vs. recall).
- **Guiding Improvements:** Identifying areas where the model needs improvement (e.g., reducing false positives).

## Types of Evaluation Metrics
Evaluation metrics vary depending on the type of machine learning task. In this directory, we focus on metrics for **classification** problems, where the goal is to assign inputs to discrete categories.

### 1. Accuracy
- **Definition:** Proportion of correct predictions among all predictions.
- **Best for:** Balanced datasets where all classes are equally important.
- **Limitation:** Can be misleading for imbalanced datasets.

### 2. Precision
- **Definition:** Proportion of positive predictions that are actually correct.
- **Best for:** Situations where the cost of false positives is high (e.g., medical diagnosis, spam detection).
- **Limitation:** Ignores false negatives.

### 3. Recall
- **Definition:** Proportion of actual positive cases that are correctly identified.
- **Best for:** Situations where missing positive cases is costly (e.g., disease screening, fraud detection).
- **Limitation:** Ignores false positives.

### 4. F1 Score
- **Definition:** Harmonic mean of precision and recall, providing a single metric that balances both.
- **Best for:** Imbalanced datasets or when both false positives and false negatives are important.
- **Limitation:** Ignores true negatives.

## Overview of Included Evaluation Metrics
This directory contains detailed tutorials and examples for the following metrics:

### 1. [Accuracy](01_Accuracy/README.md)
- **Formula:** $\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Use Cases:** Balanced classification, quick model evaluation.
- **Advantages:** Simple, intuitive, widely used.
- **Disadvantages:** Misleading for imbalanced data, does not distinguish error types.
- **[See details and Python example.](01_Accuracy/README.md)**

### 2. [Precision](02_Precision/README.md)
- **Formula:** $\mathrm{Precision} = \frac{TP}{TP + FP}$
- **Use Cases:** Information retrieval, medical diagnosis.
- **Advantages:** Focuses on positive class, reduces false positives.
- **Disadvantages:** Ignores false negatives, may not reflect overall performance.
- **[See details and Python example.](02_Precision/README.md)**

### 3. [Recall](03_Recall/README.md)
- **Formula:** $\mathrm{Recall} = \frac{TP}{TP + FN}$
- **Use Cases:** Medical screening, fraud detection.
- **Advantages:** Focuses on positive cases, complements precision.
- **Disadvantages:** Ignores false positives, may not reflect overall performance.
- **[See details and Python example.](03_Recall/README.md)**

### 4. [F1 Score](04_F1_Score/README.md)
- **Formula:** $\mathrm{F1\ Score} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$
- **Use Cases:** Imbalanced datasets, model selection.
- **Advantages:** Balances precision and recall, robust to class imbalance.
- **Disadvantages:** Ignores true negatives, may not reflect overall performance.
- **[See details and Python example.](04_F1_Score/README.md)**

## How to Choose an Evaluation Metric
- **Balanced Data:** Use accuracy for a quick, overall measure.
- **Imbalanced Data:** Prefer precision, recall, or F1 score.
- **High Cost of False Positives:** Focus on precision.
- **High Cost of False Negatives:** Focus on recall.
- **Need for Balance:** Use F1 score.
- **Multiple Metrics:** Often, it is best to consider several metrics together for a complete picture.

## Further Reading and Examples
Each subfolder contains a detailed explanation and a Python example for the respective metric. Explore them for mathematical details, use cases, advantages/disadvantages, and code implementations:
- [01_Accuracy](01_Accuracy/README.md)
- [02_Precision](02_Precision/README.md)
- [03_Recall](03_Recall/README.md)
- [04_F1_Score](04_F1_Score/README.md)

---

**References:**
- [Wikipedia: Evaluation metrics](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Deep Learning Book: Evaluation](https://www.deeplearningbook.org/contents/ml.html)
