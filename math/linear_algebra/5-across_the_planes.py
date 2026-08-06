#!/usr/bin/env python3
"""Module that adds two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element-wise.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.

    Returns:
        list: A new matrix with the element-wise sums, or None if
        ``mat1`` and ``mat2`` do not have the same shape.
    """
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    return [[a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
