#!/usr/bin/env python3
"""Module that performs element-wise operations on two matrices."""


def np_elementwise(mat1, mat2):
    """Perform element-wise addition, subtraction, multiplication
    and division.

    Args:
        mat1: A numpy.ndarray or a scalar. Never empty.
        mat2: A numpy.ndarray or a scalar. Never empty.

    Returns:
        tuple: The element-wise sum, difference, product and quotient.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
