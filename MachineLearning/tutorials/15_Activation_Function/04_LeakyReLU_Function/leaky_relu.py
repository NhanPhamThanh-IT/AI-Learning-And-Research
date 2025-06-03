import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
  """
  Leaky ReLU activation function
  """
  return np.maximum(alpha * x, x)

# Illustrate the Leaky ReLU function
x = np.linspace(-10, 10, 100)
y_leaky_relu = leaky_relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y_leaky_relu, label='Leaky ReLU Function (alpha=0.01)')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Leaky ReLU Function')
plt.grid(True)
plt.legend()
plt.show()