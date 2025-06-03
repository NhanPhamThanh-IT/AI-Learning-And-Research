# Word2Vec (Skip-gram and CBOW)

## Introduction
**Word2Vec** is a group of shallow, two-layer neural network models designed to generate word embeddings. Developed by a team at Google, the Word2Vec models are highly efficient at learning high-quality word embeddings from large text corpora. These embeddings are dense vectors that capture semantic and syntactic relationships between words, allowing for impressive results in tasks like word analogy.

The core idea behind Word2Vec is the distributional hypothesis, which states that words that appear in similar contexts have similar meanings.

Word2Vec includes two main model architectures:

1.  **Skip-gram:** Given a target word, the model tries to predict the context words within a defined window.
2.  **CBOW (Continuous Bag of Words):** Given the context words within a defined window, the model tries to predict the target word.

## Skip-gram Model
In the Skip-gram model, the input is the target word (represented as a one-hot vector), and the output layer is a softmax classifier that outputs the probability distribution over the vocabulary for context words. The model is trained to maximize the probability of observing the actual context words given the target word.

The objective function to maximize for a given target word $w_t$ and its context words $w_{t-c}, \dots, w_{t+c}$ (within a window of size $c$) is:

$$ \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} | w_t) $$

Where $P(w_o | w_i)$ is the probability of the output (context) word $w_o$ given the input (target) word $w_i$, calculated using the softmax function involving the dot product of the input and output word vectors.

Skip-gram is generally better at capturing rare words and phrases compared to CBOW.

## CBOW Model
In the CBOW model, the input is the sum or average of the one-hot vectors of the context words within a window, and the output layer is a softmax classifier that outputs the probability distribution over the vocabulary for the target word. The model is trained to maximize the probability of observing the actual target word given the context words.

The objective function to maximize for a given target word $w_t$ and its context words is:

$$ \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | \text{Context}(w_t)) $$

Where $P(w_t | \text{Context}(w_t))$ is the probability of the target word $w_t$ given its context, calculated using softmax.

CBOW is typically faster to train than Skip-gram and performs slightly better with frequent words.

## Training Word2Vec
The training process involves optimizing the word vectors to minimize a loss function (or maximize the objective function) related to predicting context from target (Skip-gram) or target from context (CBOW). Directly calculating the softmax over the entire vocabulary is computationally expensive, especially with large vocabularies. To make training more efficient, Word2Vec commonly uses approximation techniques:

- **Negative Sampling:** Instead of updating weights for all output words, only update the weights for the positive examples (the actual context words) and a small number of randomly selected negative examples (words not in the context). The objective becomes distinguishing the target word from these negative samples.
- **Hierarchical Softmax:** Uses a Huffman tree to represent the vocabulary. The probability of a word is calculated as the probability of taking a specific path from the root to the leaf node representing the word in the tree. This reduces the computation from being proportional to the vocabulary size to being proportional to the logarithm of the vocabulary size.

## Characteristics of Word2Vec Embeddings
- **Distributed Representation:** Words are represented as dense vectors where each dimension captures some latent feature.
- **Semantic and Syntactic Relationships:** Vector arithmetic can reveal relationships (e.g., 'king' - 'man' + 'woman' $\approx$ 'queen').
- **Learned from Context:** Embeddings are learned based on the contexts in which words appear.
- **Efficient Training:** Designed for efficient training on large datasets.

## Word2Vec vs GloVe
Here's a brief comparison between Word2Vec and GloVe:
- **Word2Vec (Predictive):** Learns embeddings by predicting neighboring words within a local window. Captures local context information.
- **GloVe (Count-based):** Learns embeddings by incorporating global word-word co-occurrence statistics. Captures global statistical information.

Both approaches aim to produce dense word vectors that capture semantic relationships, and often their performance is comparable. The choice might depend on the specific characteristics of the corpus and the downstream task.

## Applications
Word2Vec embeddings are foundational in many Natural Language Processing tasks:
- **Word Similarity and Relatedness:** Identifying words with similar meanings.
- **Word Analogies:** Solving analogy problems.
- **Feature Representation:** Used as input features for downstream NLP models (text classification, sentiment analysis, named entity recognition, etc.).
- **Clustering and Visualization:** Grouping similar words and visualizing relationships.

## Example
We can use the `gensim` library in Python to train and use a Word2Vec model. 