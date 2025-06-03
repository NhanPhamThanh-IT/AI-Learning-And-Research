from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

class BaggingModel:
    def __init__(self, n_estimators=10):
        self.model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 