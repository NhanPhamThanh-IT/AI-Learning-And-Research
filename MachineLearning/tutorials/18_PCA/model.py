from sklearn.decomposition import PCA

class PCAModel:
    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)
        self.explained_variance_ratio_ = None
    def fit_transform(self, X):
        X_pca = self.model.fit_transform(X)
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        return X_pca 