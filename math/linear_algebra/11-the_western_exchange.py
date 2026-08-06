#!/usr/bin/env python3
"""Module that transposes a numpy.ndarray."""


def np_transpose(matrix):
    """Return the transpose of a numpy.ndarray.

    Args:
        matrix: A numpy.ndarray.

    Returns:
        numpy.ndarray: A new array that is the transpose of ``matrix``.
    """
    return matrix.transpose()
