#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

# ## Numpy
# 
# __Numpy__ is a library provides support for large, multi-dimensional
# arrays and matrices, along with a large collection of high-level
# mathematical functions to operate on these arrays
# 
# pip3 install numpy
# 
# We need to import the library before using it

# #### 1-D arrays
# 
# - numpy.array([value1, value2, ...])
#   - All elements are of the same type
#   - The type of the array will be numpy.ndarray
#   - Type of the data:  var.dtype
#   - Size of the array (number of elements): var.size
#   - Number of array dimensions: var.ndim
#   - Shape: var.shape

import numpy as np

np_array = np.array([1, 3, -2, 2, 5, -3, -5])
print(np_array)
print("Type: {}".format(type(np_array)))
print("Element type: {}".format(np_array.dtype))
print("Size (num of el): {}".format(np_array.size))
print("ndim (num of dimensions): {}".format(np_array.ndim))
print("shape: {0}".format(np_array.shape))


# #### Indexing and slicing
# 
# - var_array[i]: To get element in position i
# - var_array[m:n]: Element from position n to m-1
# - var_array[m:n] = value1, value2, ... : To assign values to that
# slice of the array
# - var_array[[list of positions]] = value1, value2, ...: Same as above
# but applied to the positions in the list

# var_array[i]
print(np_array[1])


# var_array[m: n]
print(np_array[2:6])


# var_array[m: n] = value1, value2, value3
np_array[2:6] = 5, 4, 3, 2
print(np_array)


# var_array[list of positions] = value1, value2, ...
np_array[[1, 3, 5]] = -2, -4, -6
print(np_array)


print(np_array)


# #### Basic operations
# 
# - Addition:
#   - z = u + v
#   - numpy.add(u, v)

u = np.array([0, 1, 2])
v = np.array([2, 1, 0])
z = u + v
print(z)

u1 = np.array([0, 1, 2])
v1 = np.array([2, 1, 0])
z1 = np.add(u1, v1)
print(z1)


# - Addition of a constant:
#   - z = u + n

u = np.array([0, 1, 2])
z = u + 4
print(z)


# - Subtraction:
#   - z = u - v
#   - numpy.subtract(u, v)

u = np.array([0, 1, 2])
v = np.array([2, 1, 0])
z = u - v
print(z)

u1 = np.array([0, 1, 2])
v1 = np.array([2, 1, 0])
z1 = np.subtract(u1, v1)
print(z1)


# - Multiplication (by scalar):
#   - z = u * n


u = np.array([0, 1, 2])
z = u * 2
print(z)


# - Product of two arrays:
#   - z = u * v
#   - z = numpy.multiply(u, v)

u = np.array([0, 1, 2])
v = np.array([2, 1, 0])
z = u * v
print(z)

u1 = np.array([0, 1, 2])
v1 = np.array([2, 1, 0])
z1 = np.multiply(u1, v1)
print(z1)


# - Dot product:
#   - z = numpy.__dot__(u, v)


u = np.array([0, 1, 2])
v = np.array([2, 1, 0])
z = np.dot(u, v)
print(z)


# - Division of two arrays:
#   - z = u / v
#   - z = numpy.divide(u, v)


u = np.array([1, 2, 3])
v = np.array([3, 2, 1])
z = u / v
print(z)


u = np.array([1, 2, 3])
v = np.array([3, 2, 1])
z = np.divide(u, v)
print(z)


# #### Universal functions
# 
# - Mean: u.__mean__()
# - Min value: u.__min__()
# - Max value: u.__max__()
# - Standard Deviation: u.__std__()


u_mean = u.mean()
print("Mean of array u: {0}".format(u_mean))

u_min = u.min()
print("Min value in array u: {0}".format(u_min))

u_max = u.max()
print("Max value in array u: {0}".format(u_max))

u_std = u.std()
print("Std of array u: {0}".format(u_std))


# #### Utils
# 
# - linspace: Returns evenly spaced number over an internal:
# numpy.__linspace__(start, end, n_elements)
# - numpy.__pi__
# - numpy.__sin__(x)

evenly_spaced = np.linspace(5, 25, 5)
print(evenly_spaced)


# #### 2-D arrays (matrices)
# 
# - Creation: var_mat = numpy.__array__([row1, row2, row3, ...]):
# Rows are arrays
# - Indexing: var_mat[__i__][__j__]: i -> row and j -> col
# - Indexing alt.: var_mat[__i, j__]
# - Slicing: var_mat[x, m:n]
# - Multiplication: numpy.dot(var_mat1, var_mat2)
# - Transpose matrix: var_mat.__T__

# Creation

var_row1 = np.array([0, 1, 2])
var_row2 = np.array([3, 4, 5])
var_row3 = np.array([6, 7, 8])

mat_var = np.array([var_row1, var_row2, var_row3])
print(mat_var)

# Indexing

print(mat_var[0][0])
print(mat_var[0, 1])


# Slicing

print(mat_var[0:2, 0:2])

# Dot product

var_row12 = np.array([0, 1, 0])
var_row22 = np.array([1, 0, 1])
var_row32 = np.array([0, 1, 0])

mat_var2 = np.array([var_row12, var_row22, var_row32])

mat_res = np.dot(mat_var, mat_var2)
print(mat_res)

# Transpose matrix

print(mat_var.T)

# Plotting Mathematical Functions
# 
# import __matplotlib.pyplot__ as plt
# 
# In Jupyter we can say:
# 
# % __matplotlib__ inline
# 
# plt.plot(x, y) where x and y are vectors


x_axis = np.array([0, 1, 2])
y_axis = np.array([2, 4, 6])
plt.plot(x_axis, y_axis)
