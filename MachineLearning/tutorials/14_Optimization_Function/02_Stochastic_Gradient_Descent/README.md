# Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is an approximation of Gradient Descent. Instead of computing the gradient of the entire dataset, SGD computes the gradient of a single randomly selected training example at each step. This makes SGD much faster than standard Gradient Descent for large datasets.

## Algorithm Steps

1. Initialize the parameters (weights and biases) randomly.
2. Choose a learning rate (alpha) and a number of epochs (an epoch is one pass through the entire dataset).
3. For each epoch:
   a. Shuffle the training data.
   b. For each training example in the dataset:
      i. Compute the gradient of the cost function with respect to the parameters using *only* the current example.
      ii. Update the parameters using the formula: `parameters = parameters - learning_rate * gradient`
4. Repeat until the number of epochs is reached.

## Key Concepts

- **Stochastic**: Refers to the randomness introduced by picking one example at a time.
- **Epoch**: One complete pass through the entire training dataset.
- **Learning Rate Schedule**: Often, the learning rate is decreased over time (e.g., linearly or exponentially) to help the algorithm converge.

## Advantages

- Much faster than Batch Gradient Descent for large datasets.
- Can escape shallow local minima due to the noisy updates.

## Disadvantages

- The updates are noisy, causing the cost function to oscillate around the minimum.
- May not converge to the exact minimum, but rather a region around it.
- Can be sensitive to the learning rate schedule.

## Files in this Directory

- `README.md`: This file.
- `data.py`: Contains code for generating or loading sample data.
- `model.py`: Defines the function to be minimized (e.g., a simple cost function or a linear model) and its gradient for a single example.
- `main.py`: Demonstrates the SGD algorithm in action. 