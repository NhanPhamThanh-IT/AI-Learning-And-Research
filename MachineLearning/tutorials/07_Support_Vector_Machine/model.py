from sklearn.svm import SVC

class SVM:
    def __init__(self, kernel='linear'):
        self.clf = SVC(kernel=kernel)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean() 