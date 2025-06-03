import numpy as np

class SimpleMultinomialNaiveBayes:
    """
    A simple Multinomial Naive Bayes classifier for text classification.
    Assumes features (word counts) follow a Multinomial distribution.
    Uses Laplace smoothing.
    """

    def fit(self, X, y):
        """
        Trains the Multinomial Naive Bayes model.

        Args:
            X (np.ndarray): Training features (word count matrix).
            y (np.ndarray): Training labels.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1] # Vocabulary size

        self.class_priors = {}
        self.word_counts_by_class = {}
        self.total_words_by_class = {}

        # Calculate class priors and word counts per class
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / len(y)
            self.word_counts_by_class[c] = np.sum(X_c, axis=0) # Sum of word counts for each word in class c
            self.total_words_by_class[c] = np.sum(self.word_counts_by_class[c]) # Total words in class c

    def predict_proba(self, X):
        """
        Calculates the log probabilities for each class for the given samples.

        Args:
            X (np.ndarray): Test features (word count matrix).

        Returns:
            dict: A dictionary with class labels as keys and log probabilities as values.
        """
        log_probabilities = {}

        for c in self.classes:
            # Start with the log prior
            log_prob_c = np.log(self.class_priors[c])

            # Calculate log likelihoods: log(P(word | class)) for each word in the sample
            # Using Laplace smoothing: (count(word, class) + 1) / (total_words_in_class + vocab_size)
            smoothed_probs = (self.word_counts_by_class[c] + 1) / (self.total_words_by_class[c] + self.n_features)
            log_smoothed_probs = np.log(smoothed_probs)

            # Multiply (in log space, add) by the word counts in the sample
            log_likelihood = np.sum(X * log_smoothed_probs, axis=1) # Sum across features for each sample

            log_probabilities[c] = log_prob_c + log_likelihood

        return log_probabilities

    def predict(self, X):
        """
        Predicts the class labels for new data.

        Args:
            X (np.ndarray): Test features (word count matrix).

        Returns:
            np.ndarray: Predicted labels.
        """
        log_probabilities = self.predict_proba(X)

        # Find the class with the highest log probability for each sample
        predictions = []
        for i in range(X.shape[0]):
            max_log_prob = -np.inf
            predicted_class = None
            for c in self.classes:
                if log_probabilities[c][i] > max_log_prob:
                    max_log_prob = log_probabilities[c][i]
                    predicted_class = c
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
