#!/usr/bin/env python3
'''
This function def np_cat
that concatenates two matrices along a specific axis
assuming that mat1 and mat2 can be interpreted
as numpy.ndarrays
it must return a new numpy.ndarray
it is not allowed to use any loops or conditional statements
it may use: import numpy as np
assuming that mat1 and mat2 are never empty
'''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''
    This function concatenates two matrices
    '''
    return np.concatenate((mat1, mat2), axis=axis)
