import numpy as np

class SimpleComplementNaiveBayes:
    """
    A simple Complement Naive Bayes classifier for text classification.
    Models the probability of a document *not* belonging to a class.
    Uses Laplace smoothing (alpha=1).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha # Laplace smoothing parameter

    def fit(self, X, y):
        """
        Trains the Complement Naive Bayes model.

        Args:
            X (np.ndarray): Training features (e.g., word count matrix).
            y (np.ndarray): Training labels.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1] # Vocabulary size

        self.class_priors = {}
        self.complement_word_counts = {}
        self.total_complement_words = {}
        self.complement_feature_probs = {} # P(wi | ~C)

        total_samples = len(y)
        total_word_count = np.sum(X)

        # Calculate statistics for the complement of each class
        for c in self.classes:
            # Find indices of samples NOT belonging to class c
            complement_indices = np.where(y != c)[0]
            X_complement = X[complement_indices]
            y_complement = y[complement_indices]

            # Prior probability of the complement class P(~C)
            n_complement_samples = len(complement_indices)
            self.class_priors[c] = n_complement_samples / total_samples # This is P(~C) for class c

            # Total word count in the complement class
            self.total_complement_words[c] = np.sum(X_complement)

            # Word counts for each feature in the complement class
            self.complement_word_counts[c] = np.sum(X_complement, axis=0)

            # Calculate P(wi | ~C) using Laplace smoothing
            # P(wi | ~C) = (count(wi, ~C) + alpha) / (N_~C + alpha * V)
            self.complement_feature_probs[c] = (self.complement_word_counts[c] + self.alpha) / \
                                               (self.total_complement_words[c] + self.alpha * self.n_features)

    def predict_proba(self, X):
        """
        Calculates the log probabilities of belonging to the complement of each class
        for the given samples (log P(~C | d)).

        Args:
            X (np.ndarray): Test features (e.g., word count matrix).

        Returns:
            dict: A dictionary with original class labels as keys and log probabilities
                  of the *complement* class as values.
        """
        log_complement_posteriors = {}

        for c in self.classes:
            # Log prior of the complement class: log(P(~C))
            log_complement_prior = np.log(self.class_priors[c])

            # Log likelihood: log(P(d | ~C)) = sum(log(P(wi | ~C)) for wi in d)
            # This is sum over features: sum(count(wi, d) * log(P(wi | ~C)))

            # Ensure log probabilities are calculated from the smoothed probabilities
            log_complement_feature_probs = np.log(self.complement_feature_probs[c])

            # Calculate log likelihood for each sample
            # X is shape (n_samples, n_features)
            # log_complement_feature_probs is shape (n_features,)
            # log_likelihoods is shape (n_samples,)
            log_likelihoods = np.sum(X * log_complement_feature_probs, axis=1)

            # Log posterior: log(P(~C | d)) = log(P(~C)) + log(P(d | ~C))
            log_complement_posteriors[c] = log_complement_prior + log_likelihoods

        return log_complement_posteriors

    def predict(self, X):
        """
        Predicts the class labels for new data using the Complement Naive Bayes rule.
        Predicts the class C that minimizes P(~C | d).

        Args:
            X (np.ndarray): Test features (e.g., word count matrix).

        Returns:
            np.ndarray: Predicted labels.
        """
        log_complement_posteriors = self.predict_proba(X)

        # Find the class C that minimizes log P(~C | d) for each sample
        predictions = []
        sample_indices = range(X.shape[0])

        for i in sample_indices:
            min_log_complement_prob = np.inf # Initialize with positive infinity
            predicted_class = None

            for c in self.classes:
                if log_complement_posteriors[c][i] < min_log_complement_prob:
                    min_log_complement_prob = log_complement_posteriors[c][i]
                    predicted_class = c

            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y) 