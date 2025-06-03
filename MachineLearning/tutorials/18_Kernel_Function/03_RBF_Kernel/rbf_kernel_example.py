import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Generate non-linearly separable data (two interleaved half circles)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Scale the features - RBF kernel is sensitive to feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train an SVM model with RBF Kernel
# Experiment with different gamma and C values
model = svm.SVC(kernel='rbf', gamma=0.5, C=1.0) # Example parameters
model.fit(X_scaled, y)

print("--- RBF SVM Training Complete ---")
print(f"Number of support vectors for class 0: {model.n_support_[0]}")
print(f"Number of support vectors for class 1: {model.n_support_[1]}")
print(f"Total number of support vectors: {model.n_support_.sum()}")
print(f"Gamma: {model._gamma}")
print(f"C: {model.C}")

# 3. Visualize the decision boundary

# Create a mesh to plot the decision boundary
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training points
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

# Plot support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, 
            facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.title('SVM with RBF Kernel')
plt.legend()
plt.show() 