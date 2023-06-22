# Scientific Computing using Python – High Performance Computing in Python (2023)






## Software Design and Considerations
This section underlines the key considerations about the software design and implementation. The software is implemented in Python 3.8. The software is designed to be used as a module. No classes were implemented in this project, instead all functionality is implemented as functions. All necessary functionality for calculating the mandelbrot set is implemented in the `mandelbrot` module. The module contains the following:
- `mandelbrot.py`: Contains the main function for calculating the mandelbrot set.
- `mandelbrot_multiprocessing.py`: Contains a function for calculating the mandelbrot set using multiprocessing.

The `mandelbrot` module is documented using docstrings. The docstrings are formatted according to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). The docstrings can be viewed using the `help` function in Python. E.g. `help(mandelbrot.calculate_mandelbrot_set)`.

The design of the software is based on the following considerations:
- Functions should be grouped by functionality in modules.
- Functions should be documented using docstrings.
- Functions should be tested using unit tests with the `pytest` package.
- Functions are named as verb-noun pairs in lower case, e.g. `calculate_mandelbrot_set`.
- Functions should have a single responsibility. E.g. a function that calculates the mandelbrot set should not also be responsible for plotting the set.
- Functions should have as few parameters as possible.
- All function parameters should be passed as arguments to avoid global variables.
- Function input and output type should be declared in the function.



## Testing
All functions are tested using unit tests. The unit tests are implemented using the `pytest` package and are located in the `tests` folder. The unit tests are documented with the GIVEN/WHEN/THEN pattern. An example of such a pattern could be:
```python
"""
GIVEN: The input to the function.
WHEN: The function is called.
THEN: The expected output of the function.
"""
```

To run the unit tests, simply run:
```bash
pytest
```

## Performance and optimization
The `src/profile.py` will run a profile of functions in the `mandelbrot` module with the `kernprof -l -v src/profile.py` command.   

```
Total time: 391.049 s
File: /workspaces/scientific_computing_in_python/Scientific_Computing_Mandelbrot/mandelbrot/mandelbrot.py
Function: calculate_mandelbrot_naive at line 60

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    60                                           @profile
    61                                           def calculate_mandelbrot_naive(c: complex, max_iters: int = 100) -> float:
    62                                               """The naive implementation of the Mandelbrot set.
    63                                           
    64                                               This function is the naive implementation of the Mandelbrot set. The naive approach is to disregard any vectorization and
    65                                               instead take a single value, c, and iterate until it escapes or the maximum number of iterations is reached.
    66                                               It is to be used with enumerate_mandelbrot_set
    67                                           
    68                                           
    69                                               Args:
    70                                                   c:
    71                                                       A complex number.
    72                                                   max_iters:
    73                                                       The maximum number of iterations.
    74                                           
    75                                               Returns:
    76                                                   The number of iterations it took to escape.
    77                                           
    78                                               Examples:
    79                                                   calculate_mandelbrot_naive(-2 + 1j, 100)
    80                                           
    81                                               """
    82                                               # Initialize z and c
    83  25000000    3234580.7      0.1      0.8      z = 0
    84                                           
    85                                               # Iterate until the maximum number of iterations is reached
    86 524452924   81548704.1      0.2     20.9      for i in range(max_iters):
    87                                                   # Calculate the next value of z
    88 524452924  147446648.4      0.3     37.7          z = z**2 + c
    89                                           
    90                                                   # Check if z has escaped
    91 503747230  153278314.6      0.3     39.2          if abs(z) > 2:
    92  20705694    4908624.0      0.2      1.3              return (i + 1) / max_iters
    93                                           
    94                                               # If z has not escaped, return the maximum number of iterations
    95   4294306     632337.2      0.1      0.2      return 1
```

```
Total time: 36.3712 s
Function: calculate_mandelbrot_vectorized at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def calculate_mandelbrot_vectorized(C: np.ndarray, max_iters: int = 100) -> np.ndarray:
   101                                               """the vectorized implementation of the Mandelbrot set.
   102                                           
   103                                               This version is the vectorized implementation of the Mandelbrot set. It uses numpy arrays and operations to
   104                                               calculate the Mandelbrot set for each element in C.
   105                                           
   106                                               Args:
   107                                                   C:
   108                                                       A 2D array of complex numbers.
   109                                                   max_iters:
   110                                                       The maximum number of iterations.
   111                                           
   112                                               Returns:
   113                                                   A 2D array of the number of iterations it took to escape.
   114                                           
   115                                               Examples:
   116                                                   C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])
   117                                                   calculate_mandelbrot_vectorized(C, 100)
   118                                           
   119                                               """
   120                                               # Initialize z and c
   121         1         22.8     22.8      0.0      z = np.zeros(C.shape, np.complex128)
   122         1          5.5      5.5      0.0      mandelbrotSet = np.zeros(C.shape, np.float64)
   123         1       3345.3   3345.3      0.0      escaped = np.zeros_like(mandelbrotSet, dtype=bool)
   124                                           
   125                                               # Iterate until the maximum number of iterations is reached
   126       100         98.3      1.0      0.0      for i in range(max_iters):
   127                                                   # Calculate the next value of z only for those elements where z has not escaped yet. The ~ is a bitwise NOT i.e it flips true to false and vice versa
   128       100     290140.5   2901.4      0.8          mask = ~escaped
   129       100    7888983.1  78889.8     22.3          z[mask] = z[mask] ** 2 + C[mask]
   130                                           
   131                                                   # Check if z has escaped
   132       100   22166665.4 221666.7     62.7          escaped_this_iter = np.abs(z) >= 2
   133                                           
   134                                                   # Update the mandelbrot set
   135       100    3959990.8  39599.9     11.2          mandelbrotSet = np.where(
   136       100     609398.4   6094.0      1.7              escaped_this_iter & ~escaped, (i + 1) / max_iters, mandelbrotSet
   137                                                   )
   138                                                   # Update the escaped array. The | is a bitwise OR i.e. if either of the elements is true, the result is true
   139       100     400439.4   4004.4      1.1          escaped = escaped | escaped_this_iter
   140                                           
   141         1      52151.8  52151.8      0.1      mandelbrotSet = np.where(mandelbrotSet == 0, 1, mandelbrotSet)
   142         1          0.4      0.4      0.0      return mandelbrotSet
```


```
Total time: 25.4289 s
Function: calculate_mandelbrot_vectorized_optim at line 145

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   145                                           @profile
   146                                           def calculate_mandelbrot_vectorized_optim(
   147                                               C: np.ndarray, max_iters: int = 100
   148                                           ) -> np.ndarray:
   149                                               """the vectorized implementation of the Mandelbrot set.
   150                                           
   151                                               This version is the vectorized implementation of the Mandelbrot set. It uses numpy arrays and operations to
   152                                               calculate the Mandelbrot set for each element in C.
   153                                           
   154                                               Args:
   155                                                   C:
   156                                                       A 2D array of complex numbers.
   157                                                   max_iters:
   158                                                       The maximum number of iterations.
   159                                           
   160                                               Returns:
   161                                                   A 2D array of the number of iterations it took to escape.
   162                                           
   163                                               Examples:
   164                                                   C = np.array([[-2 + 1j, -2 + 1j], [-2 + 1j, -2 + 1j]])
   165                                                   calculate_mandelbrot_vectorized(C, 100)
   166                                           
   167                                               """
   168                                               # Initialize z and c
   169         1         19.0     19.0      0.0      z = np.zeros(C.shape, np.complex128)
   170         1          5.5      5.5      0.0      mandelbrotSet = np.zeros(C.shape, np.float64)
   171         1        923.8    923.8      0.0      escaped = np.zeros_like(mandelbrotSet, dtype=bool)
   172                                           
   173                                               # Iterate until the maximum number of iterations is reached
   174       100         83.1      0.8      0.0      for i in range(max_iters):
   175                                                   # Calculate the next value of z only for those elements where z has not escaped yet. The ~ is a bitwise NOT i.e it flips true to false and vice versa
   176       100     289655.6   2896.6      1.1          mask = ~escaped
   177       100    7825834.0  78258.3     30.8          z[mask] = z[mask] ** 2 + C[mask]
   178                                           
   179                                                   # Check if z has escaped
   180       100   12271922.7 122719.2     48.3          escaped_this_iter = (z.real**2 + z.imag**2) >= 4
   181                                           
   182                                                   # Update the mandelbrot set
   183       100    3968136.9  39681.4     15.6          mandelbrotSet = np.where(
   184       100     612949.2   6129.5      2.4              escaped_this_iter & ~escaped, (i + 1) / max_iters, mandelbrotSet
   185                                                   )
   186                                                   # Update the escaped array. The | is a bitwise OR i.e. if either of the elements is true, the result is true
   187       100     402588.1   4025.9      1.6          escaped = escaped | escaped_this_iter
   188                                           
   189         1      56751.4  56751.4      0.2      mandelbrotSet = np.where(mandelbrotSet == 0, 1, mandelbrotSet)
   190         1          0.3      0.3      0.0      return mandelbrotSet
```