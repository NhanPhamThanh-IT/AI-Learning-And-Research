# Bernoulli Naive Bayes

This directory contains a basic implementation and example of the Bernoulli Naive Bayes classification algorithm.

## What is Bernoulli Naive Bayes?

Bernoulli Naive Bayes is a variation of the Naive Bayes algorithm specifically designed for binary features, where each feature indicates the presence or absence of something (e.g., a word in a document). It models the occurrence of these binary features using the Bernoulli distribution.

## Mathematics Behind Bernoulli Naive Bayes

The foundation of Bernoulli Naive Bayes is Bayes' Theorem, combined with the naive assumption of feature independence given the class. For binary features $x_i \in \{0, 1\}$, the conditional probability $P(x_i | y)$ is modeled using the Bernoulli distribution. The formula for this conditional probability is:

$$P(x_i|y) = P(i|y)x_i + (1-P(i|y))(1-x_i)$$

Where:
* $P(i|y)$ is the probability of feature $i$ being present (having a value of 1) given class $y$.
* If $x_i = 1$, $P(x_i|y) = P(i|y)$.
* If $x_i = 0$, $P(x_i|y) = 1-P(i|y)$.

To estimate $P(i|y)$ from the training data, Laplace smoothing is typically applied to avoid zero probabilities for unseen feature-class combinations. The smoothed probability is calculated as:

$$P(i|y) = \frac{\text{count}(x_i=1, y) + \alpha}{\text{count}(y) + \alpha \cdot 2}$$

Where:
* $\text{count}(x_i=1, y)$ is the number of training samples in class $y$ where feature $i$ is present (value is 1).
* $\text{count}(y)$ is the total number of training samples in class $y$.
* $\alpha$ is the smoothing parameter (commonly 1 for Laplace smoothing).
* The denominator includes $\alpha \cdot 2$ because for a binary feature, there are two possible outcomes (0 or 1).

Bayes' Theorem is then used to calculate the posterior probability for each class $y$ given a binary feature vector $X$:

$$P(y|X) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i | y)$$

Where $P(y)$ is the prior probability of class $y$ and $P(x_i | y)$ is calculated using the Bernoulli conditional probability formula.

## Bernoulli Distribution

The Bernoulli distribution is a discrete probability distribution for a random variable that takes the value 1 with probability $p$ and the value 0 with probability $1-p$. Its probability mass function (PMF) is:

$$f(x; p) = p^x (1-p)^{1-x} \quad \text{for } x \in \{0, 1\}$$

This distribution is fundamental to Bernoulli Naive Bayes as it models the probability of a binary feature being present or absent.

## Example: Spam Classification

Consider the spam classification example from the article:

| Message ID | Message Text        | Class    |
| ---------- | ------------------- | -------- |
| M1         | "buy cheap now"     | Spam     |
| M2         | "limited offer buy" | Spam     |
| M3         | "meet me now"       | Not Spam |
| M4         | "let's catch up"    | Not Spam |

**Binary Feature Matrix (Presence = 1, Absence = 0):**

| ID | buy | cheap | now | limited | offer | meet | me | let's | catch | up | Class    |
| -- | --- | ----- | --- | ------- | ----- | ---- | -- | ----- | ----- | -- | -------- |
| M1 | 1   | 1     | 1   | 0       | 0     | 0    | 0  | 0     | 0     | 0  | Spam     |
| M2 | 1   | 0     | 0   | 1       | 1     | 0    | 0  | 0     | 0     | 0  | Spam     |
| M3 | 0   | 0     | 1   | 0       | 0     | 1    | 1  | 0     | 0     | 0  | Not Spam |
| M4 | 0   | 0     | 0   | 0       | 0     | 0    | 0  | 1     | 1     | 1  | Not Spam |

**Applying Laplace Smoothing (with $\alpha = 1$):**

The formula for $P(i|y) = \frac{\text{count}(x_i=1, y) + 1}{\text{count}(y) + 2}$ is used. Here, $\text{count}(y) = 2$ for both classes.

**Word Probabilities (P(word=1 | Class)):**

* **Spam Class:**
  * $P(\text{buy}=1 | \text{Spam}) = \frac{2+1}{2+2} = \frac{3}{4} = 0.75$
  * $P(\text{cheap}=1 | \text{Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{now}=1 | \text{Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{limited}=1 | \text{Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{offer}=1 | \text{Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{meet}=1 | \text{Spam}) = \frac{0+1}{2+2} = \frac{1}{4} = 0.25$
  * (and similarly for other words not in Spam class)

* **Not Spam Class:**
  * $P(\text{now}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{meet}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{me}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{let's}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{catch}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * $P(\text{up}=1 | \text{Not Spam}) = \frac{1+1}{2+2} = \frac{2}{4} = 0.5$
  * (and similarly for other words not in Not Spam class)

**Classifying Message "buy now":**

The binary feature vector for "buy now" (assuming all words from the vocabulary are features) is $\{\text{buy}: 1, \text{cheap}: 0, \text{now}: 1, \text{limited}: 0, \text{offer}: 0, \text{meet}: 0, \text{me}: 0, \text{let's}: 0, \text{catch}: 0, \text{up}: 0\}$.

Assume equal prior probabilities $P(\text{Spam}) = P(\text{Not Spam}) = 0.5$.

* **For Spam:**
  * $P(\text{Spam} | \text{buy now}) \propto P(\text{Spam}) \cdot P(\text{buy}=1 | \text{Spam}) \cdot P(\text{cheap}=0 | \text{Spam}) \cdots P(\text{now}=1 | \text{Spam}) \cdots$
  * $P(\text{Spam} | \text{buy now}) \propto 0.5 \cdot P(\text{buy}=1 | \text{Spam}) \cdot P(\text{cheap}=0 | \text{Spam}) \cdots P(\text{now}=1 | \text{Spam}) \cdots$
  * Using $P(x_i=0|y) = 1-P(i|y)$:
  * $P(\text{Spam} | \text{buy now}) \propto 0.5 \cdot P(\text{buy}=1 | \text{Spam}) \cdot (1-P(\text{cheap}=1 | \text{Spam})) \cdots P(\text{now}=1 | \text{Spam}) \cdots$
  * $P(\text{Spam} | \text{buy now}) \propto 0.5 \cdot 0.75 \cdot (1-0.5) \cdot 0.5 \cdots \cdot (1-0.25) \cdots$

* **For Not Spam:**
  * $P(\text{Not Spam} | \text{buy now}) \propto P(\text{Not Spam}) \cdot P(\text{buy}=1 | \text{Not Spam}) \cdot (1-P(\text{cheap}=1 | \text{Not Spam})) \cdots P(\text{now}=1 | \text{Not Spam}) \cdots$
  * $P(\text{Not Spam} | \text{buy now}) \propto 0.5 \cdot 0.25 \cdot (1-0.25) \cdot 0.5 \cdots \cdot (1-0.5) \cdots$

Comparing the proportional probabilities, the message "buy now" is classified as **Spam**.

## Advantages of Bernoulli Naive Bayes

* Suitable for binary features.
* Simple and computationally efficient.
* Can perform well in text classification where the presence/absence of words is more important than their frequency.

## Disadvantages of Bernoulli Naive Bayes

* Assumes feature independence, which is often not true.
* Less effective if feature counts are important (Multinomial Naive Bayes might be better).

## Applications of Bernoulli Naive Bayes

* Text classification (spam detection, document classification) focusing on word presence.
* Any classification task with binary features.