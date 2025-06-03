import numpy as np

class SimpleBernoulliNaiveBayes:
    """
    A simple Bernoulli Naive Bayes classifier for binary features.
    Uses Laplace smoothing (alpha=1).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha # Laplace smoothing parameter

    def fit(self, X, y):
        """
        Trains the Bernoulli Naive Bayes model.

        Args:
            X (np.ndarray): Training features (binary, 0 or 1).
            y (np.ndarray): Training labels.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1] # Number of features

        self.class_priors = {}
        self.feature_prob = {} # P(xi=1 | y)
        self.feature_prob_not = {} # P(xi=0 | y)

        # Calculate class priors and feature probabilities
        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c) # Number of samples in class c
            self.class_priors[c] = n_c / len(y)

            # Calculate P(xi=1 | y) using Laplace smoothing
            # count(xi=1, y) is the number of times feature i is 1 in class c
            count_xi_1_c = np.sum(X_c, axis=0)
            # P(xi=1 | y) = (count(xi=1, y) + alpha) / (count(y) + alpha * 2)
            self.feature_prob[c] = (count_xi_1_c + self.alpha) / (n_c + self.alpha * 2)
            # P(xi=0 | y) = 1 - P(xi=1 | y)
            self.feature_prob_not[c] = 1.0 - self.feature_prob[c]

    def predict_proba(self, X):
        """
        Calculates the log probabilities for each class for the given samples.

        Args:
            X (np.ndarray): Test features (binary).

        Returns:
            dict: A dictionary with class labels as keys and log probabilities (posteriors) as values.
        """
        log_posteriors = {}

        for c in self.classes:
            # Start with the log prior
            log_prior_c = np.log(self.class_priors[c])

            # Calculate log likelihoods for each feature for each sample
            # log(P(X | y)) = sum( log(P(xi | y)) for i in features )
            # log(P(xi | y)) = log(P(xi=1 | y)) if xi=1, log(P(xi=0 | y)) if xi=0
            # log(P(xi | y)) = xi * log(P(xi=1 | y)) + (1-xi) * log(P(xi=0 | y))

            # Use the pre-calculated log probabilities for features
            log_feature_prob_c = np.log(self.feature_prob[c])
            log_feature_prob_not_c = np.log(self.feature_prob_not[c])

            # Calculate log likelihood for each sample
            # X is shape (n_samples, n_features)
            # log_feature_prob_c and log_feature_prob_not_c are shape (n_features,)
            # The log likelihood for a sample is the sum of log probabilities for each feature
            # For a sample X[j, :], log_likelihood_j = sum(X[j, i] * log_feature_prob_c[i] + (1-X[j, i]) * log_feature_prob_not_c[i] for i in features)

            # This can be done efficiently using matrix multiplication/element-wise operations:
            log_likelihoods = np.sum(X * log_feature_prob_c + (1 - X) * log_feature_prob_not_c, axis=1)

            # Calculate log posterior for each sample: log(P(y|X)) = log(P(y)) + log(P(X|y))
            log_posteriors[c] = log_prior_c + log_likelihoods

        return log_posteriors

    def predict(self, X):
        """
        Predicts the class labels for new data.

        Args:
            X (np.ndarray): Test features (binary).

        Returns:
            np.ndarray: Predicted labels.
        """
        log_posteriors = self.predict_proba(X)

        # Find the class with the highest log posterior for each sample
        predictions = []
        # log_posteriors is a dictionary where keys are class labels and values are arrays of log posteriors for each sample
        # We need to iterate through samples (index i) and find the class with max log posterior at that index
        sample_indices = range(X.shape[0])
        for i in sample_indices:
            max_log_prob = -np.inf
            predicted_class = None
            for c in self.classes:
                if log_posteriors[c][i] > max_log_prob:
                    max_log_prob = log_posteriors[c][i]
                    predicted_class = c
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)