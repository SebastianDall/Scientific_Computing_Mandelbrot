import multiprocessing
from .mandelbrot import calculate_mandelbrot_vectorized
import numpy as np


# define the workers
def worker(args: tuple) -> np.ndarray:
    """Define the worker function for multiprocessing

    This function is used to define the worker function for multiprocessing. It takes a tuple of arguments to be used in the
    calculation of the Mandelbrot set and returns the result of the calculation.

    Args:
        args:
            A tuple of arguments to be used in the calculation of the Mandelbrot set.
            (C_slice, max_iters)

    Returns:
        A 2D array of the number of iterations it took to escape.


    """

    C_slice, max_iters = args
    return calculate_mandelbrot_vectorized(C_slice, max_iters)


def calculate_mandelbrot_multithreaded(
    C: np.ndarray, max_iters: int, cpus: int
) -> np.ndarray:
    """Multithreaded implementation for calculating the Mandelbrot set


    This function calculates the Mandelbrot set for a given array of complex numbers. It uses the multiprocessing module
    to calculate the Mandelbrot set in parallel using the calculate_mandelbrot_vectorized function. The number of CPUs
    declared are used to slice the array of complex numbers into equal parts and calculate the Mandelbrot set for each.

    Args:
        C:
            A 2D array of complex numbers
        max_iters:
            The maximum number of iterations
        cpus:
            The number of CPUs to use

    Returns:
        A 2D array of the number of iterations it took to escape


    Raises:
        ValueError:
            If cpus is not in the range 1 to the number of CPUs available on the system

    Examples:
        calculate_mandelbrot_multithreaded(np.array([[-2, 2], [-2, 2]]), 100, 2)

    """

    # create the pool
    if cpus not in range(1, multiprocessing.cpu_count() + 1):
        raise ValueError(f"cpus must be in 1 to {multiprocessing.cpu_count()}")

    pool = multiprocessing.Pool(cpus)

    # slice the array of complex numbers into equal parts
    slice_indices = [
        (i * C.shape[0] // cpus, (i + 1) * C.shape[0] // cpus) for i in range(cpus)
    ]

    # calculate the mandelbrot set for each slice
    results = pool.map(
        worker,
        [
            (C[slice_rows[0] : slice_rows[1], :], max_iters)
            for slice_rows in slice_indices
        ],
    )

    # # Close the pool
    pool.close()
    pool.join()

    # # Join results
    return np.concatenate(results)
