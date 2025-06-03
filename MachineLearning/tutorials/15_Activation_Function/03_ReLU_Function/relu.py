import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  """
  ReLU activation function
  """
  return np.maximum(0, x)

# Illustrate the ReLU function
x = np.linspace(-10, 10, 100)
y_relu = relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y_relu, label='ReLU Function')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('ReLU Function')
plt.grid(True)
plt.legend()
plt.show()