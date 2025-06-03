import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  """
  Sigmoid activation function
  """
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  """
  Derivative of the Sigmoid function
  """
  s = sigmoid(x)
  return s * (1 - s)

# Minh họa hàm Sigmoid và đạo hàm của nó
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)
y_sigmoid_derivative = sigmoid_derivative(x)

plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid Function')
plt.plot(x, y_sigmoid_derivative, label='Sigmoid Derivative', linestyle='--')

plt.xlabel('x')
plt.ylabel('Output')
plt.title('Sigmoid Function and its Derivative')
plt.grid(True)
plt.legend()
plt.show()