#!/usr/bin/env python3
"""Module that transposes a 2D matrix."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix.

    Args:
        matrix: A 2D list of ints/floats. Never empty.

    Returns:
        list: A new matrix that is the transpose of ``matrix``.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
