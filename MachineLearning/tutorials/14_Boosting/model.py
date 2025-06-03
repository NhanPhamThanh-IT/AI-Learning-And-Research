from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class BoostingModel:
    def __init__(self, n_estimators=50):
        self.model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 