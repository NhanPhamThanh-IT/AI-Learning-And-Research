# Bayes' Theorem

## Introduction
Bayes' Theorem is a fundamental result in probability theory that describes how to update the probability of a hypothesis based on new evidence. It is the foundation of Bayesian inference and is widely used in statistics, machine learning, and data science. Bayes' Theorem provides a principled way to combine prior knowledge with observed data.

## Theoretical Background
Bayes' Theorem relates the conditional and marginal probabilities of random events. For events \( A \) and \( B \):

\[
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
\]

where:
- \( P(A \mid B) \): Posterior probability of \( A \) given \( B \)
- \( P(B \mid A) \): Likelihood of \( B \) given \( A \)
- \( P(A) \): Prior probability of \( A \)
- \( P(B) \): Marginal probability of \( B \)

### Interpretation
- **Prior (P(A))**: Initial belief about \( A \) before seeing \( B \)
- **Likelihood (P(B|A))**: Probability of observing \( B \) if \( A \) is true
- **Posterior (P(A|B))**: Updated belief about \( A \) after observing \( B \)

## Types of Bayesian Methods
- **Naive Bayes Classifier**: Assumes features are independent given the class. Used for text classification, spam detection, etc.
- **Bayesian Networks**: Graphical models representing dependencies among variables.
- **Hierarchical Bayesian Models**: Allow for modeling of complex, multi-level data structures.

## Applications
Bayes' Theorem is used in:
- Naive Bayes classifiers (text classification, spam filtering)
- Medical diagnosis (updating disease probability given test results)
- Machine learning models involving uncertainty
- Risk assessment and decision making
- Genetics and bioinformatics

## Example
Suppose a disease affects 1% of a population. A test detects the disease with 99% accuracy, but also has a 5% false positive rate. If a person tests positive, what is the probability they actually have the disease?

Let:
- \( D \): Has disease
- \( T \): Tests positive

\[
P(D|T) = \frac{P(T|D)P(D)}{P(T)}
\]

Plugging in the numbers:
- \( P(D) = 0.01 \)
- \( P(T|D) = 0.99 \)
- \( P(T|\neg D) = 0.05 \)
- \( P(\neg D) = 0.99 \)
- \( P(T) = P(T|D)P(D) + P(T|\neg D)P(\neg D) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0594 \)
- \( P(D|T) = \frac{0.99 \times 0.01}{0.0594} \approx 0.167 \)

So, even with a positive test, the probability of having the disease is only about 16.7%.

## Limitations and Considerations
- Requires accurate prior probabilities, which may be subjective
- Can be computationally intensive for complex models
- Sensitive to the choice of prior
- Assumes independence in Naive Bayes, which may not hold in practice

## Comparison with Frequentist Methods
- Bayesian methods incorporate prior knowledge and update beliefs with data
- Frequentist methods rely solely on observed data
- Bayesian inference provides a full probability distribution over parameters, not just point estimates

## References
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Gelman, A., et al. (2013). Bayesian Data Analysis. CRC Press. 