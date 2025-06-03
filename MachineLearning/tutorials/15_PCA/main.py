from data import load_data
from model import PCAModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = load_data()
    model = PCAModel(n_components=2)
    X_pca = model.fit_transform(X)
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
    # Visualization for 2D PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.title('PCA Result (First 2 Principal Components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show() 