from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 