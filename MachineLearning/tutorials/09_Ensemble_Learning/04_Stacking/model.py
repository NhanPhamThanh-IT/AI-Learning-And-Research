from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class StackingModel:
    def __init__(self):
        self.model = StackingClassifier(
            estimators=[
                ('lr', LogisticRegression()),
                ('dt', DecisionTreeClassifier()),
                ('svc', SVC(probability=True))
            ],
            final_estimator=LogisticRegression(),
            passthrough=False
        )
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 