# Complement Naive Bayes

This directory contains a basic implementation and example of the Complement Naive Bayes (CNB) classification algorithm.

## What is Complement Naive Bayes?

Complement Naive Bayes (CNB) is a variant of the Naive Bayes algorithm that is particularly effective for text classification problems with imbalanced datasets. Unlike standard Naive Bayes classifiers which model the probability of a document belonging to a class $C$, CNB models the probability of a document *not* belonging to class $C$ (i.e., belonging to the complement of $C$, denoted as $\bar{C}$). It then assigns the document to the class $C$ for which $P(\bar{C} | d)$ is minimized.

## How Does Complement Naive Bayes Work?

CNB calculates the probability of a document $d$ belonging to the complement of a class $C$, $P(\bar{C} | d)$. Using Bayes' Theorem and the naive independence assumption, this can be expressed as:

$$P(\bar{C}|d) \propto P(\bar{C}) \cdot \prod_{i=1}^{n} P(w_i|\bar{C})^{f_i}$$

Where:
* $P(\bar{C})$ is the prior probability of the complement of class $C$.
* $P(w_i|\bar{C})$ is the conditional probability of word $w_i$ appearing in a document *not* belonging to class $C$.
* $f_i$ is the frequency (count) of word $w_i$ in document $d$.

To estimate $P(w_i|\bar{C})$, CNB considers all documents that are *not* in class $C$. Laplace smoothing is typically applied:

$$P(w_i | \bar{C}) = \frac{\text{count}(w_i, \bar{C}) + \alpha}{N_{\bar{C}} + \alpha \cdot V}$$

Where:
* $\text{count}(w_i, \bar{C})$ is the number of times word $w_i$ appears in all training documents *not* belonging to class $C$.
* $N_{\bar{C}}$ is the total number of words in all training documents *not* belonging to class $C$.
* $V$ is the size of the vocabulary.
* $\alpha$ is the Laplace smoothing parameter (commonly 1).

After calculating $P(\bar{C} | d)$ for each class $C$, the document $d$ is assigned to the class $C$ that minimizes this probability:

$$\hat{C} = \arg\min_{C} P(\bar{C}|d)$$

This is equivalent to maximizing $P(C|d)$, but by focusing on the complement, CNB tends to be more robust to class imbalance, as the statistics for the complement class $\bar{C}$ are generally more stable than those for a small minority class $C$.

## Advantages of Complement Naive Bayes

* Effective for imbalanced datasets, which is a common challenge in text classification.
* Simple and computationally efficient, similar to other Naive Bayes variants.
* Less affected by the distribution of the minority class.

## Disadvantages of Complement Naive Bayes

* Still relies on the naive independence assumption.
* May not perform as well as more complex models on balanced datasets.

## Applications of Complement Naive Bayes

* Text classification with imbalanced classes (e.g., spam detection where spam is a minority class).
* Document categorization in scenarios where some categories have significantly fewer examples than others.