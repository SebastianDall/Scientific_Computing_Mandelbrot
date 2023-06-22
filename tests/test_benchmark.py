import time
import pytest
from mandelbrot.benchmark import benchmark_functions


def test_benchmark_functions():
    """
    GIVEN a list of functions, where the only function is time.sleep(1)
    WHEN the benchmark function is used
    THEN the result should be a list with one element, which is the name of the function and the time it took to execute
    """

    functions = [("wait", lambda: time.sleep(1))]

    times = benchmark_functions(functions)
    assert times[0][0] == "wait"
    pytest.approx(times[0][1], 1, 0.1)
