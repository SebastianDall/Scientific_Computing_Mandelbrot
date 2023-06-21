import multiprocessing
from .mandelbrot import calculate_mandelbrot_vectorized
import numpy as np


# define the workers
def worker(args):
    C_slice, max_iters = args
    return calculate_mandelbrot_vectorized(C_slice, max_iters)


def calculate_mandelbrot_multithreaded(C, max_iters, cpus):
    """
    This function calculates the Mandelbrot set using multiple threads

    The Mandelbrot is calculated with the vectorized implementation

    :param C: A 2D array of complex numbers
    :param max_iters: The maximum number of iterations
    :param cpus: The number of CPUs to use
    :return: A 2D array of the number of iterations it took to escape
    """

    # create the pool
    if cpus not in range(1, multiprocessing.cpu_count() + 1):
        raise ValueError(f"cpus must be in {range(1, multiprocessing.cpu_count() + 1)}")

    pool = multiprocessing.Pool(cpus)

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
