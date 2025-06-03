# GloVe (Global Vectors for Word Representation)

## Introduction
**GloVe** is an unsupervised learning algorithm for obtaining vector representations for words. Developed by researchers at Stanford, GloVe aims to capture semantic relationships between words based on global word-word co-occurrence statistics collected from a large corpus. Unlike purely predictive models like Word2Vec that learn embeddings based on local context windows, GloVe incorporates global statistical information.

## How GloVe Works
GloVe's approach is based on the idea that the ratio of word-word co-occurrence probabilities has meaning. Consider two words, 'ice' and 'steam', and a set of probe words like 'solid', 'gas', 'water', and 'fashion'. The ratio of the probability that 'ice' co-occurs with a probe word to the probability that 'steam' co-occurs with the same probe word can tell us a lot about the relationship between 'ice' and 'steam'.

GloVe trains word vectors such that their dot product relates to the logarithm of their co-occurrence probability. The model minimizes a cost function that penalizes the difference between the dot product of two word vectors and the logarithm of their co-occurrence count:

$$ J = \sum_{i=1}^{V} \sum_{j=1}^{V} f(X_{ij}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2 $$

Where:
- $V$ is the size of the vocabulary.
- $X_{ij}$ is the number of times word $i$ co-occurs with word $j$.
- $\mathbf{w}_i$ and $\tilde{\mathbf{w}}_j$ are the word vectors for words $i$ and $j$ (GloVe learns two sets of vectors, often summed or averaged in the end).
- $b_i$ and $\tilde{b}_j$ are bias terms.
- $f(X_{ij})$ is a weighting function that gives less weight to very frequent co-occurrences (which might be less informative) and zero weight to zero co-occurrences.

The weighting function $f(x)$ is typically defined as:

$$ f(x) = \begin{cases} (x/x_{\max})^{\alpha} & \text{if } x < x_{\max} \\ 1 & \text{if } x \ge x_{\max} \end{cases} $$

Where $x_{\max}$ and $\alpha$ are hyperparameters (e.g., $x_{\max} = 100$ and $\alpha = 0.75$). This function caps the weight for very frequent co-occurrences.

The objective function encourages words that co-occur frequently to have a larger dot product (and thus be closer in the vector space) and captures relationships through ratios of co-occurrence probabilities.

## Characteristics
- **Global Statistics:** Leverages global co-occurrence counts across the entire corpus.
- **interpretable Vector Differences:** Word analogy relationships (e.g., 'king' - 'man' + 'woman' $\approx$ 'queen') are often well-captured as vector differences.
- **Pre-trained Embeddings Available:** Large pre-trained GloVe vectors (trained on Wikipedia, Common Crawl, etc.) are publicly available and widely used.
- **Fixed Vocabulary:** Requires a fixed vocabulary determined from the corpus.

## GloVe vs Word2Vec
Both GloVe and Word2Vec are popular techniques for learning word embeddings, but they differ in their approach:
- **Word2Vec (Skip-gram/CBOW):** Predictive models that learn embeddings based on local context windows. Focuses on predicting neighboring words.
- **GloVe:** A count-based model that leverages global co-occurrence statistics from the entire corpus. Focuses on the ratios of co-occurrence probabilities.

In practice, both often produce high-quality embeddings, and the choice between them can depend on the specific task and dataset.

## Applications
GloVe embeddings are widely used in various Natural Language Processing tasks:
- **Word Similarity:** Measuring semantic similarity between words.
- **Word Analogies:** Solving analogy tasks based on vector arithmetic.
- **Input Features:** Used as initial word embeddings in neural network models for tasks like text classification, named entity recognition, machine translation, etc.
- **Text Classification and Clustering:** Representing documents as aggregated word vectors.

## Using Pre-trained GloVe Embeddings
One of the most common ways to use GloVe is by downloading pre-trained vectors. These vectors are trained on massive text corpora and capture rich semantic information. You can load these vectors into a lookup table or an embedding layer in a neural network.

## Example
We can demonstrate how to load and use pre-trained GloVe embeddings in Python. 