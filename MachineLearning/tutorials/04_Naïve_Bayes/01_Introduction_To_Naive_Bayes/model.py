import numpy as np

class SimpleNaiveBayes:
    """
    A simple Naive Bayes classifier for categorical features.
    """

    def fit(self, X, y):
        """
        Trains the Naive Bayes model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.class_priors = {}
        self.conditional_probabilities = {}

        # Calculate class priors
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / len(y)

        # Calculate conditional probabilities P(xi | y)
        for i in range(self.n_features):
            self.conditional_probabilities[i] = {}
            for c in self.classes:
                # Get feature values for the current class
                X_c = X[y == c, i]
                # Calculate probabilities for each unique feature value
                unique_values, counts = np.unique(X_c, return_counts=True)
                self.conditional_probabilities[i][c] = dict(zip(unique_values, counts / len(X_c)))

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
                # Add log likelihoods (conditional probabilities)
                for i in range(self.n_features):
                    feature_value = sample[i]
                    # Get the conditional probability, handle unseen values with smoothing if necessary
                    # For simplicity here, we assume all test feature values were seen in training
                    # A more robust implementation would use Laplace smoothing or similar.
                    prob_xi_given_y = self.conditional_probabilities[i][c].get(feature_value, 1e-9) # Add small epsilon for unseen values
                    log_posterior += np.log(prob_xi_given_y)
                posteriors[c] = log_posterior

            # Predict the class with the highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
