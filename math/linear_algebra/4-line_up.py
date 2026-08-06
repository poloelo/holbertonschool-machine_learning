#!/usr/bin/env python3
"""Module that adds two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Add two 1D arrays element-wise.

    Args:
        arr1: A list of ints/floats.
        arr2: A list of ints/floats.

    Returns:
        list: A new list with the element-wise sums, or None if
        ``arr1`` and ``arr2`` do not have the same shape.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
