#!/usr/bin/env python3
"""Module that calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Calculate the shape of a matrix.

    Args:
        matrix: A (possibly nested) list. All elements in the same
            dimension are assumed to be of the same type/shape.

    Returns:
        list: The shape of the matrix as a list of integers.
    """
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
