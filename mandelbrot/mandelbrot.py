# This function is the naive implementation of the Mandelbrot set
def mandelbrot_naive(c, max_iters):
    """
    This function is the naive implementation of the Mandelbrot set
    :param c: The complex number to check
    :param max_iters: The maximum number of iterations
    :return: The number of iterations it took to escape
    """
    # Initialize z and c
    z = 0

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iters):
        # Calculate the next value of z
        z = z**2 + c

        # Check if z has escaped
        if abs(z) > 2:
            return (i + 1) / max_iters

    # If z has not escaped, return the maximum number of iterations
    return max_iters
