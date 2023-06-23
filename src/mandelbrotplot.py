import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# set to use latex for text rendering

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mandelbrot.mandelbrot import (
    enumerate_mandelbrot_set,
    calculate_mandelbrot_naive_with_numba,
)

# Load the data
h5py_file = h5py.File("data/mandelbrot.hdf5", "r")
C = h5py_file["C"][:]
h5py_file.close()

max_iterations = 100

# Calculate the mandelbrot set
MandelbrotSet = enumerate_mandelbrot_set(
    C, max_iterations, calculate_mandelbrot_naive_with_numba
)
log_mandelbrot = np.log10(MandelbrotSet + 1e-9)

# Plot the mandelbrot set
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"  # or 'stix'
mpl.rcParams["font.family"] = "STIXGeneral"

plt.figure()
plt.imshow(log_mandelbrot, extent=[-2, 1, -1.5, 1.5])
plt.xlabel(r"$\mathrm{Re}(c)$", fontsize=16)
plt.ylabel(r"$\mathrm{Im}(c)$", fontsize=16)
plt.title("Mandelbrot Set", fontsize=20)
plt.colorbar()
plt.set_cmap("hot")
plt.savefig("figures/mandelbrotplot.png")
