import numpy as np
from mandelbrot.mandelbrot import calculate_mandelbrot_naive, enumerate_mandelbrot_set


# Test the naive implementation
def test_mandelbrot_naive():
    """
    GIVEN a normal and a complex number
    WHEN the naive implementation is used
    THEN one result should be 100 and the other should be 0.02
    """
    assert calculate_mandelbrot_naive(0, 100) == 1
    assert calculate_mandelbrot_naive(-2 + 1j, 100) == 0.01


def test_enumerate_mandelbrot_set():
    """
    GIVEN a 2D array of complex numbers
    WHEN the enumerate function is used
    THEN the result should be a 2D array of the number of iterations it took to escape
    """

    C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])

    assert np.array_equal(
        enumerate_mandelbrot_set(C, 100), np.array([[0.01, 0.01], [0.01, 0.01]])
    )
