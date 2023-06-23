import numpy as np
import h5py
import os

if not os.path.exists("./data"):
    os.makedirs("./data")

import argparse

parser = argparse.ArgumentParser(description="Create a dataset for the Mandelbrot set")
parser.add_argument("--pre", type=int, default=5000)
parser.add_argument("--pim", type=int, default=5000)

# Create an array of complex numbers
args = parser.parse_args()
pre = args.pre
pim = args.pim

pre_from = -2
pre_to = 1
pim_from = -1.5
pim_to = 1.5

R = np.linspace(pre_from, pre_to, pre)
I = 1j * np.linspace(pim_from, pim_to, pim)

R, I = np.meshgrid(R, I)

C = R + I

with h5py.File("./data/mandelbrot.hdf5", "a") as h5py_file:
    ds_name = f"C_{pre}_{pim}"
    if ds_name in h5py_file:
        del h5py_file[ds_name]  # delete the existing dataset
    h5py_file.create_dataset(ds_name, data=C)  # create the new dataset
