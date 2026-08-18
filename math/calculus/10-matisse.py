#!/usr/bin/env python3
"""Module that computes the derivative of a polynomial."""


def poly_derivative(poly):
    """Compute the derivative of a polynomial.

    Args:
        poly: A list of coefficients, where the index of the list is the
            power of x the coefficient belongs to.

    Returns:
        list: A new list of coefficients representing the derivative,
        ``[0]`` if the derivative is 0, or None if ``poly`` is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
               for c in poly):
        return None
    derivative = [c * i for i, c in enumerate(poly)][1:]
    if not derivative or all(c == 0 for c in derivative):
        return [0]
    return derivative
