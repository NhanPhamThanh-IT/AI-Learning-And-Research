import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# 1. Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.05, random_state=42)

# 2. Train a Linear SVM model
# Use the SVC classifier with a linear kernel
model = svm.SVC(kernel='linear', C=1.0) # C is the regularization parameter
model.fit(X, y)

print("--- Linear SVM Training Complete ---")
print(f"Number of support vectors for class 0: {model.n_support_[0]}")
print(f"Number of support vectors for class 1: {model.n_support_[1]}")
print(f"Total number of support vectors: {model.n_support_.sum()}")
print(f"Model intercept: {model.intercept_}")
print(f"Model coefficients: {model.coef_}")

# 3. Visualize the decision boundary

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

# Plot support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, 
            facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM with Linear Kernel')
plt.legend()
plt.show() 