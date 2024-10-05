#!/usr/bin/env python3
""" derivative """

def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    # Check if poly is a list and not empty
    if not isinstance(poly, list) or not poly:
        return None
    
    # Handle the case for constant polynomials
    if len(poly) == 1:
        return [0]
    
    derivative = []
    
    # Calculate the derivative using the power rule
    for i in range(len(poly) - 1, 0, -1):
        # Append the derivative term to the list
        derivative.append(poly[i] * i)

    return derivative[::-1]
