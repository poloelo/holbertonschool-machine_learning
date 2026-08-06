#!/usr/bin/env python3
"""Module that concatenates two arrays."""


def cat_arrays(arr1, arr2):
    """Concatenate two 1D arrays.

    Args:
        arr1: A list of ints/floats.
        arr2: A list of ints/floats.

    Returns:
        list: A new list containing ``arr1`` followed by ``arr2``.
    """
    return list(arr1) + list(arr2)
