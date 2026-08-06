#!/usr/bin/env python3
"""Module that performs matrix multiplication."""


def mat_mul(mat1, mat2):
    """Multiply two 2D matrices.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.

    Returns:
        list: A new matrix that is the product of ``mat1`` and ``mat2``,
        or None if the two matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
            for row in mat1]
