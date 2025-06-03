# L-BFGS

The Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm is an optimization algorithm in the family of quasi-Newton methods. Quasi-Newton methods use an approximation of the Hessian matrix (the matrix of second-order partial derivatives) to find the direction of descent. L-BFGS is a popular choice for large-scale problems because it does not require the storage or computation of the full Hessian matrix.

Instead of storing the dense \(n \times n\) Hessian approximation, L-BFGS approximates the inverse Hessian matrix using a limited number of past updates of the position and gradient. This significantly reduces the memory requirement, making it suitable for problems with a large number of parameters.

## Algorithm Idea

L-BFGS approximates the inverse Hessian matrix \(H_k^{-1}\) at iteration \(k\) using information from the last \(m\) updates of the parameters (\(\Delta \mathbf{w}\)) and gradients (\(\Delta \mathbf{g}\)). The search direction is then computed as \(\mathbf{d}_k = -H_k^{-1} \mathbf{g}_k\), and the parameters are updated as \(\mathbf{w}_{k+1} = \mathbf{w}_k + \alpha_k \mathbf{d}_k\), where \(\alpha_k\) is the step size typically found using a line search method.

## Key Concepts

- **Quasi-Newton Method**: Optimization methods that use an approximation of the Hessian matrix.
- **Hessian Matrix**: A square matrix of second-order partial derivatives of a function.
- **Inverse Hessian Approximation**: L-BFGS approximates the inverse of the Hessian.
- **Limited Memory**: Uses only a limited number of past updates to perform the approximation.
- **Line Search**: A technique used to determine the step size (\(\alpha_k\)) in each iteration.

## Advantages

- Often converges faster than gradient descent methods, especially for well-conditioned problems.
- Does not require explicit computation or storage of the full Hessian matrix.
- Suitable for large-scale optimization problems.

## Disadvantages

- Can be more complex to implement than gradient descent.
- The limited-memory approximation might not be accurate for all functions.
- Requires storing a certain number of past updates.

## Files in this Directory

- `README.md`: This file.
- `data.py`: Placeholder for data generation or loading.
- `model.py`: Placeholder for defining the function to be minimized and its gradient.
- `main.py`: Demonstrates how to use an L-BFGS implementation (e.g., from SciPy) to minimize a function. 