from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, max_depth=None):
        self.clf = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 