from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class VotingModel:
    def __init__(self):
        self.model = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression()),
                ('dt', DecisionTreeClassifier()),
                ('svc', SVC(probability=True))
            ],
            voting='soft'
        )
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 