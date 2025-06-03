import numpy as np

def softmax(x):
    """
    Softmax activation function

    Args:
        x: A numpy array or list of numbers.

    Returns:
        A numpy array of the same shape as x, representing a probability distribution.
    """
    # To avoid overflow, subtract the maximum value
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example usage
z = np.array([1.0, 2.0, 3.0])
probabilities = softmax(z)
print("Input vector:", z)
print("Softmax output (probabilities):", probabilities)
print("Sum of probabilities:", np.sum(probabilities))

# Another example with larger values
z2 = np.array([10.0, 20.0, 30.0])
probabilities2 = softmax(z2)
print("\nInput vector:", z2)
print("Softmax output (probabilities):", probabilities2)
print("Sum of probabilities:", np.sum(probabilities2))