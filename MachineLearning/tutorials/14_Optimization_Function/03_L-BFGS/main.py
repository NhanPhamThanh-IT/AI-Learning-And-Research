# Placeholder for main script demonstrating L-BFGS usage (e.g., using SciPy optimize) 

import numpy as np
from scipy.optimize import minimize
from model import rosenbrock, rosenbrock_grad

# This example uses SciPy's optimize.minimize with the 'L-BFGS-B' method
# for demonstration purposes, as a full L-BFGS implementation is complex.

def main():
    """
    Demonstrates using SciPy's L-BFGS optimizer on the Rosenbrock function.
    """
    # Starting point for the optimization
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2]) # Example starting point for a 5-dimensional problem
    
    print(f"Initial Function Value: {rosenbrock(x0):.4f}")
    print(f"Initial Gradient Norm: {np.linalg.norm(rosenbrock_grad(x0)):.4f}")

    # Use SciPy's minimize function with L-BFGS-B method
    # The 'L-BFGS-B' method is a bound-constrained variation of L-BFGS
    # For unconstrained optimization, 'BFGS' or 'L-BFGS-B' without bounds can be used.
    result = minimize(rosenbrock, x0, method='L-BFGS-B', jac=rosenbrock_grad)

    print("\nOptimization Results:")
    print(result)
    
    print("\nOptimized Parameters (x):", result.x)
    print(f"Final Function Value: {result.fun:.4f}")
    print(f"Final Gradient Norm: {np.linalg.norm(rosenbrock_grad(result.x)):.4f}")

if __name__ == '__main__':
    main() 