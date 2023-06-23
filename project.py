import os


# Create the dataset
## if not data file exists, create it
if os.path.isfile("./data/mandelbrot.hdf5") == False:
    print("Creating dataset...")
    os.system("python3 src/create_dataset.py")
else:
    print("Dataset already exists")

# Plot the mandelbrot set
print("Plotting the mandelbrot set...")
os.system("python3 src/mandelbrotplot.py")

# Profile the functions
print("Profiling the functions...")
os.system("kernprof -l -v src/profile.py")

# Benchmark the functions
print("Benchmarking the functions...")
os.system("python3 src/benchmark.py")
