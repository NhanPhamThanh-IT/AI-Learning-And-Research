import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        posteriors = []
        for x in X:
            probs = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c]))
                likelihood -= 0.5 * np.sum(((x - self.means[c]) ** 2) / self.vars[c])
                probs.append(prior + likelihood)
            posteriors.append(np.argmax(probs))
        return np.array(posteriors)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) 