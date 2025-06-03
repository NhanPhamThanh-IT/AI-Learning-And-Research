import numpy as np
from scipy.stats import norm

class SimpleGaussianNaiveBayes:
    """
    A simple Gaussian Naive Bayes classifier.
    Assumes features follow a Gaussian distribution within each class.
    """

    def fit(self, X, y):
        """
        Trains the Gaussian Naive Bayes model.

        Args:
            X (np.ndarray): Training features (continuous).
            y (np.ndarray): Training labels.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.class_priors = {}
        self.mean = {}
        self.variance = {}

        # Calculate class priors, mean, and variance for each feature per class
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / len(y)
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0) + 1e-9 # Add small epsilon to avoid zero variance

    def predict(self, X):
        """
        Predicts the class labels for new data.

        Args:
            X (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = []
        for sample in X:
            posteriors = {}
            for c in self.classes:
                # Start with the log prior
                log_posterior = np.log(self.class_priors[c])
                # Add log likelihoods (log of Gaussian PDF)
                # P(xi | yk) = (1 / (sigma * sqrt(2*pi))) * exp(-((xi - mu)^2 / (2 * sigma^2)))
                # log(P(xi | yk)) = -0.5 * log(2*pi*sigma^2) - ((xi - mu)^2 / (2 * sigma^2))
                log_likelihood = np.sum(norm.logpdf(sample, loc=self.mean[c], scale=np.sqrt(self.variance[c])))
                log_posterior += log_likelihood
                posteriors[c] = log_posterior

            # Predict the class with the highest posterior probability (using log probabilities)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
