'''
utils.py

A module for various general use functions

Last updated February 2016

by Trevor Arp
'''

import numpy as np
from scipy.ndimage.interpolation import shift

'''
Searches $list and returns the index of $item. Returns -1 if it can't find it.
Compares as strings
'''
def indexof(list, item):
	for i in range(len(list)):
		if list[i] == item :
			return i
	print "utils.indexof() Error: Could not locate given item: " + str(item)
	return -1
# end indexof

'''
Shift the rows of the given matrix $data horizontally by a shift given by the array $offset
Will trim off the columns that are not full of data after the shift
'''
def offset_rows(data, offset):
    rows, cols = data.shape
    val = -999999
    for i in range(rows):
        data[i,:] = shift(data[i,:], offset[i], cval=val)
    start = 0
    stop = 0
    c = []
    for j in range(cols):
        if val in data[:,j]:
            c.append(j)
    return np.delete(data,c, axis=1)
# end offset rows
