'''
utils.py

A module for various general use functions

Last updated February 2020

by Trevor Arp
All Rights Reserved
'''

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import argrelextrema

def indexof(list, item):
	'''
	Searches a list and returns the index of item. Compares as strings. Returns -1 if it can't find it.

	Args:
		list : The list to search
		item : The item to find
	'''
	for i in range(len(list)):
		if list[i] == item :
			return i
	print("warning utilities.indexof: Could not locate given item: " + str(item))
	return -1
# end indexof

# Taken from https://stackoverflow.com/questions/42656930/numpy-assignment-like-numpy-take
def put_at(inds, axis=-1, slc=(slice(None),)):
	'''
	Generated indices to place data into an array in a similar manner to how numpy.take accesses
	data, is meant to be used as a[put_at(indicies, axis)] where to access that data would be
	numpy.take(a, indicies, axis)

	Args:
		inds : The indices of the values to access.
		axis : The axis over which to select values.
		slc : The slide of that axis to select.
	'''
	return (axis<0)*(Ellipsis,) + axis*slc + (inds,) + (-1-axis)*slc
# end put_at

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

'''
Shift the rows of the given matrix $data horizontally by a shift given by the array $offset
Will trim off the columns that are not full of data after the shift
'''
def offset_rows(data, offset):
    rows, cols = data.shape
    val = -999999
    for i in range(rows):
        data[i,:] = shift(data[i,:], offset[i], cval=val)
    c = []
    for j in range(cols):
        if val in data[:,j]:
            c.append(j)
    return np.delete(data,c, axis=1)
# end offset rows

'''
Get the arguments of the maximum of a 2D array
'''
def argmax2D(d):
	ix = np.argmax(d)
	r, c = np.unravel_index(ix, d.shape)
	return r, c
# end argmax2D

'''
Get the arguments $N largest local maxima (in descending order) in an array $d

Parameter $order (default 5) is the number of points that the maxima has to be greater than to qualify
'''
def argmaxima(d, N, order=5, display_warning=True):
	local_maxes = argrelextrema(d, np.greater, order=order)
	local_maxes = local_maxes[0]
	local_max_values = d[local_maxes]
	srt = np.argsort(local_max_values)
	n = local_maxes.size
	if n >= N:
		args = np.zeros(N).astype(int)
		for i in range(N):
			args[i] = local_maxes[srt[n-1-i]]
	elif n == 0:
		if display_warning:
			print("Warning utils.argmaxima: No relative local maxima found")
		args = None
	elif n < N:
		args = np.zeros(n).astype(int)
		for i in range(n):
			args[i] = local_maxes[srt[n-1-i]]
		if display_warning:
			print("Warning utils.argmaxima: Number of maxima found is less than requested number")
	return args
# end argmaxima

'''
Get the arguments $N largest local minima (in ascending order of value) in an array $d

Parameter $order (default 5) is the number of points that the maxima has to be greater than to qualify
'''
def argminima(d, N, order=5, display_warning=True):
	local_mins = argrelextrema(d, np.less, order=order)
	local_mins = local_mins[0]
	local_min_values = d[local_mins]
	srt = np.argsort(local_min_values)
	n = local_mins.size
	if n >= N:
		args = np.zeros(N).astype(int)
		for i in range(N):
			args[i] = local_mins[srt[i]]
	elif n == 0:
		if display_warning:
			print("Warning utils.argminima: No relative local minima found")
		args = None
	elif n < N:
		args = np.zeros(n).astype(int)
		for i in range(n):
			args[i] = local_mins[srt[i]]
		if display_warning:
			print("Warning utils.argminima: Number of minima found is less than requested number")
	return args
# end argminima
