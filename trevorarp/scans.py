'''
scans.py

A module for manipulating data runs

Last updated February 2016

by Trevor Arp
'''

import numpy as np

'''
Takes a 2D image and averages it along the fast axis, returning line (x,y) data
Meant for scans where a parameter was varied along the slow axis
and other params were held constant

Takes a range xrange=(xmin, xmax) as a keyword argument, if None then the row index
will work as the x basis.

Returns xdata, ydata
'''
def slowscan_2_line(img, xrange=None):

	s = np.shape(img)
	if xrange == None:
		x = np.linspace(0,s[0]-1,s[0])
	elif isinstance(xrange, tuple) and len(xrange) == 2:
		x = np.linspace(xrange[0],xrange[1],s[0])
	else:
		print("Error slowscan_2_line: xrange must be a tuple with format (x_min, x_max)")
		return None,None
	y = np.zeros(s[0])
	for i in range(s[0]):
		y[i] = np.mean(img[i,:])
	return x,y
# end slowscan_2_line

'''
Return a linear range between the two values of the given key in the log file
$key is the key to search the log for,
$log is the log file
$N is the number of points in the basis
'''
def range_from_log(key, log, N):
	try:
		rng = log[key]
	except:
		print("Error range_from_log: Cannot read from log file")
		return None
	if isinstance(rng,tuple):
		return np.linspace(rng[0], rng[1], N)
	else:
		print("Error range_from_log: Given key doesn't return a range")
		return None
#

'''
Return a range corresponding to the range of the ranging parameter of a data cubs corresponding
to $log
'''
def cube_range_from_log(log):
	try:
		param = log['Ranging Parameter']
		N = log['Number of Scans']
		rng = log[param]
	except Exception as e:
		print("Error cube_range_from_log: Could not read cube params from log")
		print(e)
	return np.linspace(rng[0], rng[1],int(N))
#

'''
Subtracts $background from input $scan, assuming the background image is the same
size as the input scan
'''
def subtract_background(scan, background):
    scan_s = np.shape(scan)
    bg_s = np.shape(background)
    if scan_s != bg_s:
        print("Error subtract_background: Scan and background different sizes")
        print("Warning: Returned without subtraction")
        return scan
    return scan-background
# end subtract_background
