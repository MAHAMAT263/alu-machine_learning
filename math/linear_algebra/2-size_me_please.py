#!/usr/bin/env python3
"""A function that calculates the shape of a matrix"""
def matrix_shape(matrix):
    rows = matrix
    shape = []
    while len(rows) > 0:
        shape.append(len(rows))
        rows = rows[0] if isinstance(rows[0], list) else []
    return shape
