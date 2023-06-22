import numpy as np
from numba import jit
from typing import Callable, Optional

# No-op for use with profiling and test
try:

    @profile
    def f(x):
        return x

except:

    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner


@profile
def enumerate_mandelbrot_set(
    C: np.ndarray,
    max_iters: int,
    func: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> np.ndarray:
    """This function calculates mandebrot for each element in C.

    This function loops through every element in C and calculates the number of iterations it takes to escape.

    Args:
        C:
            A 2D array of complex numbers.
        max_iters:
            The maximum number of iterations.
        func:
            A naive function to use to calculate the Mandelbrot set.

    Returns:
        A 2D array of the number of iterations it took to escape.cape

    Examples:
        C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])
        enumerate_mandelbrot_set(C, 100, calculate_mandelbrot_naive)

    """

    if func is None:
        raise ValueError("func cannot be None")

    shape = C.shape
    MandelbrotSet = np.zeros((shape[0], shape[1]))

    for i, c in enumerate(C):
        for j, c in enumerate(C[i]):
            MandelbrotSet[i][j] = func(c, max_iters)

    return MandelbrotSet


@profile
def calculate_mandelbrot_naive(c: complex, max_iters: int = 100) -> float:
    """The naive implementation of the Mandelbrot set.

    This function is the naive implementation of the Mandelbrot set. The naive approach is to disregard any vectorization and
    instead take a single value, c, and iterate until it escapes or the maximum number of iterations is reached.
    It is to be used with enumerate_mandelbrot_set


    Args:
        c:
            A complex number.
        max_iters:
            The maximum number of iterations.

    Returns:
        The number of iterations it took to escape.

    Examples:
        calculate_mandelbrot_naive(-2 + 1j, 100)

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


@profile
def calculate_mandelbrot_vectorized(C: np.ndarray, max_iters: int = 100) -> np.ndarray:
    """the vectorized implementation of the Mandelbrot set.

    This version is the vectorized implementation of the Mandelbrot set. It uses numpy arrays and operations to
    calculate the Mandelbrot set for each element in C.

    Args:
        C:
            A 2D array of complex numbers.
        max_iters:
            The maximum number of iterations.

    Returns:
        A 2D array of the number of iterations it took to escape.

    Examples:
        C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])
        calculate_mandelbrot_vectorized(C, 100)

    """
    # Initialize z and c
    z = np.zeros(C.shape, np.complex128)
    mandelbrotSet = np.zeros(C.shape, np.float64)
    escaped = np.zeros_like(mandelbrotSet, dtype=bool)

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z only for those elements where z has not escaped yet. The ~ is a bitwise NOT i.e it flips true to false and vice versa
        mask = ~escaped
        z[mask] = z[mask] ** 2 + C[mask]

        # Check if z has escaped
        escaped_this_iter = np.abs(z) >= 2

        # Update the mandelbrot set
        mandelbrotSet = np.where(
            escaped_this_iter & ~escaped, (i + 1) / max_iters, mandelbrotSet
        )
        # Update the escaped array. The | is a bitwise OR i.e. if either of the elements is true, the result is true
        escaped = escaped | escaped_this_iter

    mandelbrotSet = np.where(mandelbrotSet == 0, 1, mandelbrotSet)
    return mandelbrotSet


def calculate_mandelbrot_vectorized_optim(
    C: np.ndarray, max_iters: int = 100
) -> np.ndarray:
    """the vectorized implementation of the Mandelbrot set.

    This version is the vectorized implementation of the Mandelbrot set. It uses numpy arrays and operations to
    calculate the Mandelbrot set for each element in C.

    Args:
        C:
            A 2D array of complex numbers.
        max_iters:
            The maximum number of iterations.

    Returns:
        A 2D array of the number of iterations it took to escape.

    Examples:
        C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])
        calculate_mandelbrot_vectorized(C, 100)

    """
    # Initialize z and c
    z = np.zeros(C.shape, np.complex128)
    mandelbrotSet = np.zeros(C.shape, np.float64)
    escaped = np.zeros_like(mandelbrotSet, dtype=bool)

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z only for those elements where z has not escaped yet. The ~ is a bitwise NOT i.e it flips true to false and vice versa
        mask = ~escaped
        z[mask] = z[mask] ** 2 + C[mask]

        # Check if z has escaped
        escaped_this_iter = (z.real**2 + z.imag**2) >= 4

        # Update the mandelbrot set
        mandelbrotSet = np.where(
            escaped_this_iter & ~escaped, (i + 1) / max_iters, mandelbrotSet
        )
        # Update the escaped array. The | is a bitwise OR i.e. if either of the elements is true, the result is true
        escaped = escaped | escaped_this_iter

    mandelbrotSet = np.where(mandelbrotSet == 0, 1, mandelbrotSet)
    return mandelbrotSet


# This function is jit version of the naive implementation of the Mandelbrot set
@jit(nopython=True)
def calculate_mandelbrot_naive_with_numba(c: complex, max_iters: int = 100) -> float:
    """jit version of the naive implementation of the Mandelbrot set.

    This function is the naive implementation of the Mandelbrot set with numba. The naive approach is to disregard any vectorization and
    instead take a single value, c, and iterate until it escapes or the maximum number of iterations is reached. This function is the same as calculate_mandelbrot_naive
    It is to be used with enumerate_mandelbrot_set.

    Args:
        c:
            A complex number.
        max_iters:
            The maximum number of iterations.

    Returns:
        The number of iterations it took to escape.

    Examples:
        calculate_mandelbrot_naive_with_numba(-2 + 1j, 100)
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
