# 0x00. Linear Algebra

## Description

This project is an introduction to linear algebra applied to machine learning, using plain Python and NumPy. It covers manipulating arrays and matrices (slicing, shape, transpose, element-wise operations, concatenation, matrix multiplication) both by hand and with NumPy, as a foundation for the math used throughout the ML curriculum.

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- pycodestyle 2.11.1
- All files must be executable
- All files must end with a new line
- The first line of all files must be exactly `#!/usr/bin/env python3`
- All modules, classes, and functions must have a documentation string
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`

## Files

| File | Description |
|---|---|
| `0-slice_me_up.py` | Slicing a 1D array (first two, last five, 2nd–6th elements) |
| `1-trim_me_down.py` | Slicing a 2D matrix (3rd and 4th columns) |
| `2-size_me_please.py` | `matrix_shape(matrix)` — shape of a nested matrix |
| `3-flip_me_over.py` | `matrix_transpose(matrix)` — transpose of a 2D matrix |
| `4-line_up.py` | `add_arrays(arr1, arr2)` — element-wise addition of two arrays |
| `5-across_the_planes.py` | `add_matrices2D(mat1, mat2)` — element-wise addition of two 2D matrices |
| `6-howdy_partner.py` | `cat_arrays(arr1, arr2)` — concatenation of two arrays |
| `7-gettin_cozy.py` | `cat_matrices2D(mat1, mat2, axis=0)` — concatenation of two 2D matrices along an axis |
| `8-ridin_bareback.py` | `mat_mul(mat1, mat2)` — matrix multiplication |
| `9-let_the_butcher_slice_it.py` | NumPy slicing (middle rows, middle columns, bottom-right square) |
| `10-ill_use_my_scale.py` | `np_shape(matrix)` — shape of a `numpy.ndarray` |
| `11-the_western_exchange.py` | `np_transpose(matrix)` — transpose of a `numpy.ndarray` |
| `12-bracin_the_elements.py` | `np_elementwise(mat1, mat2)` — element-wise `+ - * /` |
| `13-cats_got_your_tongue.py` | `np_cat(mat1, mat2, axis=0)` — NumPy concatenation |
| `14-saddle_up.py` | `np_matmul(mat1, mat2)` — NumPy matrix multiplication |

## Usage

Each task is testable via its associated `X-main.py` file:

```bash
./0-main.py
```

## Author

Paul — Holberton School
