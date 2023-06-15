from mandelbrot.mandelbrot import mandelbrot_naive


# Test the naive implementation
def test_naive():
    """
    GIVEN a normal and a complex number
    WHEN the naive implementation is used
    THEN one result should be 100 and the other should be 0.02
    """
    assert mandelbrot_naive(0, 100) == 100
    assert mandelbrot_naive(-2 + 1j, 100) == 0.01
