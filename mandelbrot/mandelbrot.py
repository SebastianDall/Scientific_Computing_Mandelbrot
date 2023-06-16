import numpy as np
from numba import jit


def enumerate_mandelbrot_set(C, max_iters, func=None):
    """
    This function calculates mandebrot for each element in C
    :param C: A 2D array of complex numbers
    :param max_iters: The maximum number of iterations
    :param func: The function to use to calculate the Mandelbrot set
    :return: A 2D array of the number of iterations it took to escape
    """

    if func is None:
        raise ValueError("func cannot be None")

    shape = C.shape
    MandelbrotSet = np.zeros((shape[0], shape[1]))

    for i, c in enumerate(C):
        for j, c in enumerate(C[i]):
            MandelbrotSet[i][j] = func(c, max_iters)

    return MandelbrotSet


# This function is the naive implementation of the Mandelbrot set
def calculate_mandelbrot_naive(c, max_iters):
    """
    This function is the naive implementation of the Mandelbrot set
    :param c: The complex number to check
    :param max_iters: The maximum number of iterations
    :return: The number of iterations it took to escape
    """
    # Initialize z and c
    z = 0

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z
        z = z**2 + c

        # Check if z has escaped
        if abs(z) > 2:
            return (i + 1) / max_iters

    # If z has not escaped, return the maximum number of iterations
    return 1


# This function is the vectorized implementation of the Mandelbrot set
def calculate_mandelbrot_vectorized(C, max_iters):
    """
    This version is the vectorized implementation of the Mandelbrot set. It uses numpy arrays and operations to
    calculate the Mandelbrot set for each element in C.

    :param C: A 2D array of complex numbers
    :param max_iters: The maximum number of iterations
    :return: A 2D array of the number of iterations it took to escape
    """
    # Initialize z and c
    z = np.zeros(C.shape, np.complex128)
    mandelbrotSet = np.zeros(C.shape, np.float64)
    escaped = np.zeros_like(mandelbrotSet, dtype=bool)

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z only for those elements where z has not escaped yet
        mask = ~escaped
        z[mask] = z[mask] ** 2 + C[mask]

        # Check if z has escaped
        escaped_this_iter = np.abs(z) >= 2

        # Update the mandelbrot set
        mandelbrotSet = np.where(
            escaped_this_iter & ~escaped, (i + 1) / max_iters, mandelbrotSet
        )
        # Update the escaped array
        escaped = escaped | escaped_this_iter

    mandelbrotSet = np.where(mandelbrotSet == 0, 1, mandelbrotSet)
    return mandelbrotSet


# This function is jit version of the naive implementation of the Mandelbrot set
@jit(nopython=True)
def calculate_mandelbrot_naive_with_numba(c, max_iters):
    """
    This function is the naive implementation of the Mandelbrot set
    :param c: The complex number to check
    :param max_iters: The maximum number of iterations
    :return: The number of iterations it took to escape
    """
    # Initialize z and c
    z = 0

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z
        z = z**2 + c

        # Check if z has escaped
        if abs(z) > 2:
            return (i + 1) / max_iters

    # If z has not escaped, return the maximum number of iterations
    return 1
