import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

# 1. Generate non-linearly separable data (concentric circles)
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)

# 2. Train an SVM model with Polynomial Kernel
# Experiment with different degrees (d), gamma, and coef0 (r)
model = svm.SVC(kernel='poly', degree=3, gamma='auto', coef0=1.0, C=1.0) # Example parameters
model.fit(X, y)

print("--- Polynomial SVM Training Complete ---")
print(f"Number of support vectors for class 0: {model.n_support_[0]}")
print(f"Number of support vectors for class 1: {model.n_support_[1]}")
print(f"Total number of support vectors: {model.n_support_.sum()}")
print(f"Polynomial Kernel Degree: {model.degree}")
print(f"Gamma: {model._gamma}") # Note: _gamma is a private attribute, use model.gamma if available in future versions
print(f"Coef0: {model.coef0}")


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
plt.title('SVM with Polynomial Kernel (Degree 3)')
plt.legend()
plt.show() 