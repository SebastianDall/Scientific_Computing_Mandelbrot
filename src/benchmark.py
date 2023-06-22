import sys
import os
import h5py
import matplotlib.pyplot as plt
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mandelbrot.mandelbrot import (
    enumerate_mandelbrot_set,
    calculate_mandelbrot_naive,
    calculate_mandelbrot_vectorized,
    calculate_mandelbrot_vectorized_optim,
    calculate_mandelbrot_naive_with_numba,
)
from mandelbrot.multiprocessing_mandelbrot import calculate_mandelbrot_multithreaded

from mandelbrot.benchmark import benchmark_functions


h5py_file = h5py.File("data/mandelbrot.hdf5", "r")
C = h5py_file["C"][:]
h5py_file.close()

max_iterations = 100


# Define your functions and arguments
functions = [
    (
        "naive",
        lambda: enumerate_mandelbrot_set(C, max_iterations, calculate_mandelbrot_naive),
    ),
    ("vectorized", lambda: calculate_mandelbrot_vectorized(C, max_iterations)),
    (
        "vectorized - optim",
        lambda: calculate_mandelbrot_vectorized_optim(C, max_iterations),
    ),
    (
        "naive - numba",
        lambda: enumerate_mandelbrot_set(
            C, max_iterations, calculate_mandelbrot_naive_with_numba
        ),
    ),
    (
        "vectorized - multiprocessing, 12 threads",
        lambda: calculate_mandelbrot_multithreaded(C, max_iterations, 12),
    ),
]

# Measure execution time for each function

print(
    "Starting benchmark with C = {}x{} and {} iterations".format(
        C.shape[0], C.shape[1], max_iterations
    )
)
times = benchmark_functions(functions)


# Sort results by time
times.sort(key=lambda x: x[1], reverse=True)

# Split the tuples into two lists for easy plotting
names, elapsed_times = zip(*times)


# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(names, elapsed_times, color="steelblue")
plt.xlabel("Function")
plt.xticks(rotation=90)
plt.ylabel("Time (s)")
plt.title("Execution Time Comparison")
plt.savefig("figures/functions_benchmark.png")


functions = [
    (
        i,
        lambda i=i: calculate_mandelbrot_multithreaded(C, max_iterations, i),
    )
    for i in range(1, multiprocessing.cpu_count())
]


# Measure execution time for each function
threadsBenchmark = benchmark_functions(functions)


# Create a line, dot plot of the results
plt.figure(figsize=(10, 6))
plt.plot(
    [i[0] for i in threadsBenchmark],
    [i[1] for i in threadsBenchmark],
    "o-",
    color="steelblue",
)
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Execution Time Comparison")
plt.savefig("figures/computational_speedup.png")
