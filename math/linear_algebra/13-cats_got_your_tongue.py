#!/usr/bin/env python3
"""Module that concatenates two matrices along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy.ndarrays along a given axis.

    Args:
        mat1: A numpy.ndarray. Never empty.
        mat2: A numpy.ndarray. Never empty.
        axis: The axis along which to concatenate.

    Returns:
        numpy.ndarray: A new array, the concatenation of the inputs.
    """
    return np.concatenate((mat1, mat2), axis=axis)
