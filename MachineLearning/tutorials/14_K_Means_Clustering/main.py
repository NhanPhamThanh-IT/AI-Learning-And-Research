from data import load_data
from model import KMeansModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = load_data()
    model = KMeansModel(n_clusters=3)
    labels = model.fit_predict(X)
    print(f"Cluster centers:\n{model.cluster_centers_}")
    # Visualization for 2D data
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.title('K-Means Clustering Result')
        plt.legend()
        plt.show() 