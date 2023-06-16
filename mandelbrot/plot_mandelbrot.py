import matplotlib.pyplot as plt


def plot_mandelbrot(M, extent=None):
    """
    This function plots the Mandelbrot set

    :param M: A 2D array of the number of iterations it took to escape
    :param extent: The extent of the plot
    """

    if extent is None:
        raise ValueError("extent must be specified")

    plt.imshow(M, extent=extent)
    plt.colorbar()
    plt.show()
