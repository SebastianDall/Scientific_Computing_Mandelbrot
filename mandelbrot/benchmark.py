import timeit


def benchmark_functions(functions: list) -> list:
    """Benchmark a list of functions

    This function will iterate through a list of tuples containing the name of the function and the function itself. It will then measure the time it takes to execute each function.

    Args:
        functions:
            A list of tuples with the first element being the name of the function and the second element being the function itself.

    Returns:
        A list of tuples with the first element being the name of the function and the second element being the time it took to execute the function.

    Examples:
        benchmark_functions([("foo", lambda: print("foo"))])

    """
    times = []
    for name, func in functions:
        start_time = timeit.default_timer()
        func()
        elapsed = timeit.default_timer() - start_time
        times.append((name, elapsed))
    return times
