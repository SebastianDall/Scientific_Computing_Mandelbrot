import numpy as np
import time

import mandelbrot.mandelbrot as mb


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


methods = [
    "mb.enumerate_mandelbrot_set(C, 100, mb.calculate_mandelbrot_naive)",
    "mb.calculate_mandelbrot_vectorized(C, 100)",
    "mb.calculate_mandelbrot_vectorized_optim(C, 100)",
]

print("C = {}".format(C.shape))
for m in methods:
    y = eval(m)
    tic = time.time()
    y = eval(m)
    toc = time.time() - tic

    print("{:30s} : {:10.2e} [s]".format(m, toc))
