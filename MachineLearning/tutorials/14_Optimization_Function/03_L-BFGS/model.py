import numpy as np

def rosenbrock(x):
    """
    The Rosenbrock function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: The function value.
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    """
    The gradient of the Rosenbrock function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: The gradient vector.
    """
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    grad = np.zeros_like(x)
    grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2.0)
    grad[1:-1] = 200.0 * (xm - xm_m1**2.0) - 2.0 * (1.0 - xm) - 400.0 * xm * (xm_p1 - xm**2.0)
    grad[-1] = 200.0 * (x[-1] - x[-2]**2.0)
    return grad

if __name__ == '__main__':
    # Example usage
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    print("Rosenbrock function at x0:", rosenbrock(x0))
    print("Rosenbrock gradient at x0:", rosenbrock_grad(x0)) 