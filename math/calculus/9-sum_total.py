#!/usr/bin/env python3
"""Module that computes the sum of the first n squares."""


def summation_i_squared(n):
    """Compute the sum of i squared for i from 1 to n.

    Uses the closed form n(n + 1)(2n + 1) / 6 so that no loop is needed.

    Args:
        n: The stopping condition, a positive integer.

    Returns:
        int: The value of the sum, or None if ``n`` is not valid.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
