#!/usr/bin/env python3
"""
integral
"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial
    """
    if not isinstance(poly, list) or not poly or not isinstance(C, int):
        return None

    integral = [poly[i] / (i + 1) for i in range(len(poly) - 1, 0, -1)]
    integral.append(poly[0] / 1)
    integral.append(C)

    if len(poly) == 1 and poly[0] == 0:
        integral = [C]

    integral = [int(coeff) if coeff % 1 == 0 else coeff for coeff in integral]

    return integral[::-1]
