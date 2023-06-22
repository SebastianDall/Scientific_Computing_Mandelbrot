import numpy as np
import pytest
import multiprocessing
from mandelbrot.mandelbrot import (
    calculate_mandelbrot_naive,
    enumerate_mandelbrot_set,
    calculate_mandelbrot_vectorized,
    calculate_mandelbrot_vectorized_optim,
    calculate_mandelbrot_naive_with_numba,
)
from mandelbrot.multiprocessing_mandelbrot import calculate_mandelbrot_multithreaded


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
        enumerate_mandelbrot_set(C, 100, calculate_mandelbrot_naive),
        np.array([[0.01, 0.01], [0.01, 0.01]]),
    )


# Test the vectorized implementation
def test_mandelbrot_vectorized():
    """
    GIVEN a 2D array of complex numbers
    WHEN the vectorized implementation is used
    THEN the result should be a 2D array of the number of iterations it took to escape
    """
    C = np.array([[-2 + 1j, -2 + 1j, 0 - 0.5j], [-2 + 1j, -2 + 1j, 0 + 0.5j]])

    assert np.array_equal(
        calculate_mandelbrot_vectorized(C, 100),
        np.array([[0.01, 0.01, 1], [0.01, 0.01, 1]]),
    )


def test_mandelbrot_vectorized_optim():
    """
    GIVEN a 2D array of complex numbers
    WHEN the vectorized implementation is used
    THEN the result should be a 2D array of the number of iterations it took to escape
    """
    C = np.array([[-2 + 1j, -2 + 1j, 0 - 0.5j], [-2 + 1j, -2 + 1j, 0 + 0.5j]])

    assert np.array_equal(
        calculate_mandelbrot_vectorized_optim(C, 100),
        np.array([[0.01, 0.01, 1], [0.01, 0.01, 1]]),
    )


def test_mandelbrot_numba_naive():
    """
    GIVEN a normal and a complex number
    WHEN the naive implementation is used
    THEN one result should be 100 and the other should be 0.02
    """
    assert calculate_mandelbrot_naive_with_numba(0, 100) == 1
    assert calculate_mandelbrot_naive_with_numba(-2 + 1j, 100) == 0.01


def test_mandelbrot_multithreaded():
    """
    GIVEN a 2D array of complex numbers
    WHEN the multithreaded implementation is used
    THEN the result should be a 2D array of the number of iterations it took to escape
    """

    C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])

    assert np.array_equal(
        calculate_mandelbrot_multithreaded(C, 100, 1),
        np.array([[0.01, 0.01], [0.01, 0.01]]),
    )

    # assert error is raised when num_threads is more less than 1
    cpus_range = multiprocessing.cpu_count()
    with pytest.raises(ValueError, match=rf"cpus must be in 1 to {cpus_range}"):
        calculate_mandelbrot_multithreaded(C, 100, 0)
