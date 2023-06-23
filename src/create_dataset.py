import numpy as np
import h5py
import os

if not os.path.exists("./data"):
    os.makedirs("./data")


# Create an array of complex numbers
pre = 5000
pim = 5000

pre_from = -2
pre_to = 1
pim_from = -1.5
pim_to = 1.5

R = np.linspace(pre_from, pre_to, pre)
I = 1j * np.linspace(pim_from, pim_to, pim)

R, I = np.meshgrid(R, I)

C = R + I

h5py_file = h5py.File("./data/mandelbrot.hdf5", "w")
h5py_file.create_dataset("C", data=C)
h5py_file.close()
