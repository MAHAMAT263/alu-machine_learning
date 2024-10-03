#!/usr/bin/env python3
'''
This function def add_matrices2D
adds two matrices element-wise
assuming that mat1 and mat2 are
2D matrices containing ints/floats
assuming all elements in the
same dimension are of the same type/shape
it must return a new matrix
and If mat1 and mat2 are not the same shape, return None
'''


def add_matrices2D(mat1, mat2):
    '''
    Function computes two matrices
    of same length and return the sums element-wise
    '''
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    matrix = [[] for _ in range(len(mat1))]
    for row in range(len(mat1)):
        for col in range(len(mat1[row])):
            matrix[row].append(mat1[row][col]+mat2[row][col])
    return matrix
