# Multinomial Naive Bayes

This directory contains a basic implementation and example of the Multinomial Naive Bayes classification algorithm.

## What is Multinomial Naive Bayes?

Multinomial Naive Bayes (MNB) is a variation of the Naive Bayes algorithm particularly well-suited for classification with discrete features, such as word counts in text classification. It models the frequency of occurrences of features (e.g., words) and assumes these occurrences are multinomially distributed.

## How Does Multinomial Naive Bayes Work?

MNB works by calculating the probability of a document belonging to a specific class based on the frequency of words it contains. The "Naive" part comes from the assumption that the presence of one word is independent of the presence of other words, given the class. The "Multinomial" part refers to the multinomial distribution used to model the word counts.

The probability of a document $ d $ belonging to a class $ C $ is calculated using a form derived from Bayes' Theorem:

$$ P(C|d) \propto P(C) \cdot \prod_{i=1}^{n} P(w_i|C)^{f_i} $$

Where:
*   $ P(C|d) $ is the posterior probability of class $ C $ given document $ d $.
*   $ P(C) $ is the prior probability of class $ C $.
*   $ P(w_i|C) $ is the conditional probability of word $ w_i $ appearing in a document of class $ C $.
*   $ f_i $ is the frequency (count) of word $ w_i $ in document $ d $.

To estimate the conditional probabilities $ P(w_i|C) $, Maximum Likelihood Estimation (MLE) is often used, typically with Laplace smoothing to handle words not seen in the training data for a particular class. The formula for this smoothed probability is:

$$ \theta_{c,i} = P(w_i | C) = \frac{\text{count}(w_i, C) + 1}{N_C + V} $$

Where:
*   $ \text{count}(w_i, C) $ is the number of times word $ w_i $ appears in documents of class $ C $ in the training data.
*   $ N_C $ is the total number of words in all documents of class $ C $ in the training data.
*   $ V $ is the size of the vocabulary (total number of unique words in the training data).

## Example: Spam Classification

Let's consider the simple spam classification example from the GeeksforGeeks article. We have a small dataset of messages labeled as "Spam" or "Not Spam".

| Message ID | Message Text        | Class    |
| ---------- | ------------------- | -------- |
| M1         | "buy cheap now"     | Spam     |
| M2         | "limited offer buy" | Spam     |
| M3         | "meet me now"       | Not Spam |
| M4         | "let's catch up"    | Not Spam |

And a test message: "buy now".

1.  **Vocabulary:** The unique words form the vocabulary: $\{ \text{buy, cheap, now, limited, offer, meet, me, let's, catch, up} \}$. Vocabulary size $ V = 10 $.

2.  **Word Frequencies by Class:** Count word occurrences in each class.
    *   **Spam:** buy: 2, cheap: 1, now: 1, limited: 1, offer: 1. Total words $ N_{\text{Spam}} = 6 $.
    *   **Not Spam:** meet: 1, me: 1, now: 1, let's: 1, catch: 1, up: 1. Total words $ N_{\text{Not Spam}} = 6 $.

3.  **Apply Laplace Smoothing and Calculate Probabilities:** Using the smoothed probability formula $ P(w | C) = \frac{\text{count}(w, C) + 1}{N_C + V} $.
    *   **Spam:**
        $ P(\text{buy} | \text{Spam}) = \frac{2 + 1}{6 + 10} = \frac{3}{16} $
        $ P(\text{now} | \text{Spam}) = \frac{1 + 1}{6 + 10} = \frac{2}{16} $
    *   **Not Spam:**
        $ P(\text{buy} | \text{Not Spam}) = \frac{0 + 1}{6 + 10} = \frac{1}{16} $
        $ P(\text{now} | \text{Not Spam}) = \frac{1 + 1}{6 + 10} = \frac{2}{16} $

4.  **Calculate Posterior Probabilities for Test Message "buy now":** Assume equal prior probabilities $ P(\text{Spam}) = P(\text{Not Spam}) = 0.5 $.
    *   $ P(\text{Spam} | \text{buy now}) \propto P(\text{Spam}) \cdot P(\text{buy} | \text{Spam}) \cdot P(\text{now} | \text{Spam}) = 0.5 \cdot \frac{3}{16} \cdot \frac{2}{16} = \frac{3}{256} $
    *   $ P(\text{Not Spam} | \text{buy now}) \propto P(\text{Not Spam}) \cdot P(\text{buy} | \text{Not Spam}) \cdot P(\text{now} | \text{Not Spam}) = 0.5 \cdot \frac{1}{16} \cdot \frac{2}{16} = \frac{1}{256} $

5.  **Final Classification:** Compare the posterior probabilities.
    Since $ P(\text{Spam} | \text{buy now}) > P(\text{Not Spam} | \text{buy now}) $, the message "buy now" is classified as **Spam**.

## Advantages of Multinomial Naive Bayes

*   Simple and easy to implement.
*   Efficient, especially for large text datasets.
*   Performs well in text classification tasks.
*   Handles discrete features effectively.

## Disadvantages of Multinomial Naive Bayes

*   Relies on the naive independence assumption, which is rarely true for words in a document.
*   May not perform as well if the feature distributions are not truly multinomial.

## Applications of Multinomial Naive Bayes

*   Spam filtering.
*   Document categorization.
*   Sentiment analysis.
*   Any classification task with discrete features (e.g., counts, frequencies).
