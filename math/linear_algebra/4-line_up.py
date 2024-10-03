#!/usr/bin/env python3
'''
A function def add_arrays
which adds two arrays element-wise
assume that arr1 and arr2 are lists of ints/floats
it must return a new list
If arr1 and arr2 are not the same shape, return None
'''


def add_arrays(arr1, arr2):
    '''
    This function computes two array
    with the same length and returns the sum
    '''
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
