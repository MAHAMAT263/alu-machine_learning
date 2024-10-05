#!/usr/bin/env python3
""" derivative """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if not isinstance(poly, list) or not poly:
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(len(poly) -1, 0, -1):
        derivative.append = [poly[i] * i]
    return derivative[::-1]
