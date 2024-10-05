#!/usr/bin/env python3
""" summation """

def summation_i_squared(n):
    """ n is the stopping condition"""
    if n > 0 :
        return (n * (n + 1) * (2 * n + 1)) // 6
    else:
        return None
