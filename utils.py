'''
utils.py

A module for various general use functions

Last updated February 2016

by Trevor Arp
'''

import numpy as np
from scipy.ndimage.interpolation import shift

from datetime import date
from os.path import abspath as OS_abspath
from os.path import dirname as OS_dirname
from os.path import exists as OS_exists
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
For a standard run number returns the date at which it was taken as a datetime.date object
'''
def date_from_rn(rn):
	s = rn.split('_')
	return date(int(s[0]), int(s[1]), int(s[2]))
# end date_from_rn
