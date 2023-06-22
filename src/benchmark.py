from mandelbrot.mandelbrot import (
    enumerate_mandelbrot_set,
    calculate_mandelbrot_naive,
    calculate_mandelbrot_vectorized,
    calculate_mandelbrot_vectorized_optim,
)
import timeit
import h5py

h5py_file = h5py.File("data/mandelbrot.hdf5", "r")
C = h5py_file["C"][:]
h5py_file.close()

max_iterations = 100

# Define your functions and arguments
functions = [
    # (
    #     "naive",
    #     lambda: enumerate_mandelbrot_set(C, max_iterations, calculate_mandelbrot_naive),
    # ),
    ("vectorized", lambda: calculate_mandelbrot_vectorized(C, max_iterations)),
    (
        "vectorized - optim",
        lambda: calculate_mandelbrot_vectorized_optim(C, max_iterations),
    ),
]

# Measure execution time for each function
print(
    "Starting benchmark with C = {}x{} and {} iterations".format(
        C.shape[0], C.shape[1], max_iterations
    )
)
times = []
for name, func in functions:
    start_time = timeit.default_timer()
    func()
    elapsed = timeit.default_timer() - start_time
    times.append((name, elapsed))
