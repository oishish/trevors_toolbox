'''
utils.py

A module for various general use functions

Last updated April 2019

by Trevor Arp
'''

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from datetime import date
from os.path import abspath as OS_abspath
from os.path import dirname as OS_dirname
from os.path import exists as OS_exists
from os import makedirs
from os.path import join

'''
Searches $list and returns the index of $item. Returns -1 if it can't find it.
Compares as strings
'''
def indexof(list, item):
	for i in range(len(list)):
		if list[i] == item :
			return i
	print("utils.indexof() Error: Could not locate given item: " + str(item))
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

'''
Gets the macheine specific values of local, returns a dictionary
'''
def get_locals():
	LOCALS_dir_path = OS_dirname(OS_abspath(__file__))
	if OS_exists(join(LOCALS_dir_path, 'locals.txt')):
		f = open(join(LOCALS_dir_path, 'locals.txt'))
		out = {}
		for line in f:
			k, v = line.split('=')
			k = k.strip()
			v = v.strip()
			out[k] = v
		return out
	else:
		print("Error get_locals(): No local variables file")
#end get_locals

'''
Finds a data file by searching based on the date in the runnum, searches by looking for a log file
with the given run number.

$directory is the directory to search in, $rn is the run number

$fileend is the end of the file name, by default '_log.log'

returns the path to the directory that the files are in, or None if nothing is found
'''
def find_run(rn, directory=None, fileend='_log.log'):
	if directory is None:
		local_values = get_locals()
		directory = local_values['datadir']
	path = directory
	s = rn.split('_')
	if OS_exists(join(path, rn + fileend)):
		return path
	else:
		path = join(path, s[0])
	if OS_exists(join(path, rn + fileend)):
		return path
	else:
		path = join(path, s[0] + '_' + s[1])
	if OS_exists(join(path, rn + fileend)):
		return path
	else:
		path = join(path, s[0] + '_' + s[1] + '_' + s[2])
	if OS_exists(join(path, rn + fileend)):
		return path
	else:
		return None
# end find_run

'''
Similar to find_run but looks for the processed data savefile, if it doesn't find one it creates it
'''
def find_savefile(rn, directory=None):
	if directory is None:
		local_values = get_locals()
		directory = local_values['Processed_datadir']
	s = rn.split('_')
	path = join(directory, s[0], s[0] + '_' + s[1], s[0] + '_' + s[1] + '_' + s[2])
	if not OS_exists(path):
		makedirs(path)
	return path
# end find_run

'''
For a standard run number returns the date at which it was taken as a datetime.date object
'''
def date_from_rn(rn):
	s = rn.split('_')
	return date(int(s[0]), int(s[1]), int(s[2]))
# end date_from_rn

'''
A simple numerical derivative using numpy.gradient, computes using 2nd order central differences

- x and y are the independent and dependent variables
'''
def dydx(x, y):
	return np.gradient(y)/np.gradient(x)
# end dydx

'''
Estimate the maximum coordinates of some data from an interpolation of the data. Works best on sparse
data following a slow curve, where the maxumum is clearly between points.

Parameters:
- x,y the data to interpolate
- kind='cubic' the kind of inteprolation to use, default is cubic
- warn=True, if true will print a warning if the optimization fails

Returns:
- X0, Y0, the estimated maximum location and value.
'''
def interp_maximum(x, y, kind='cubic', warn=True):
    ifunc = interp1d(x, -1.0*y, kind=kind)
    ix = np.argmin(x)
    res = minimize_scalar(ifunc, x[ix], method='bounded', bounds=(np.min(x), np.max(x)))
    if res.success:
        x0 = res.x
    else:
        x0 = x[ix]
        if warn:
            print('Warning interp_maximum: Optimization did not exit successfully')
    return x0, -1.0*ifunc(x0)
# end interp_maximum

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
	if n < N and display_warning:
		print("Warning utils.argmaxima: Number of maxima found is less than requested number")
	args = np.zeros(N).astype(int)
	for i in range(N):
		args[i] = local_maxes[srt[n-1-i]]
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
	if n < N and display_warning:
		print("Warning utils.argminima: Number of minima found is less than requested number")
	args = np.zeros(N).astype(int)
	for i in range(N):
		args[i] = local_mins[srt[i]]
	return args
# end argminima
