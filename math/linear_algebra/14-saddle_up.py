#!/usr/bin/env python3
"""Module that performs matrix multiplication with numpy."""
import numpy as np


def np_matmul(mat1, mat2):
    """Multiply two numpy.ndarrays.

    Args:
        mat1: A numpy.ndarray. Never empty.
        mat2: A numpy.ndarray. Never empty.

    Returns:
        numpy.ndarray: The matrix product of ``mat1`` and ``mat2``.
    """
    return np.matmul(mat1, mat2)
