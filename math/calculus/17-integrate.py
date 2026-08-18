#!/usr/bin/env python3
"""Module that computes the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Compute the integral of a polynomial.

    Args:
        poly: A list of coefficients, where the index of the list is the
            power of x the coefficient belongs to.
        C: An integer representing the integration constant.

    Returns:
        list: A new list of coefficients representing the integral, as
        small as possible, or None if ``poly`` or ``C`` is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)) or isinstance(C, bool):
        return None
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
               for c in poly):
        return None
    integral = [C] + [c / (i + 1) for i, c in enumerate(poly)]
    integral = [int(c) if c == int(c) else c for c in integral]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
