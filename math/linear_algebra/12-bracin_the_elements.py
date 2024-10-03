#!/usr/bin/env python3
'''
This function def np_elementwise that performs
element-wise addition, subtraction, multiplication, and division
assuming that mat1 and mat2 can be interpreted as numpy.ndarrays
it should return a tuple containing the element-wise sum, difference,
product, and quotient, respectively
it is not allowed to use any loops or conditional statements
assuming that mat1 and mat2 are never empty
'''


def np_elementwise(mat1, mat2):
    '''
    This function return addition wises
    '''
    # addition
    sum_result = mat1 + mat2
    # subtraction
    diff_result = mat1 - mat2
    # multiplication
    product_result = mat1 * mat2
    # division
    quotient_result = mat1 / mat2
    # Return a tuple containing the results of operations
    return diff_result, sum_result, quotient_result, product_result
