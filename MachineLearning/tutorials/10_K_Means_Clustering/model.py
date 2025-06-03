from sklearn.cluster import KMeans

class KMeansModel:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers_ = None
    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        return labels 