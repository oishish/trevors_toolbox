'''
display.py

A module for general functions related to displaying information, for functions and classes related
to displaying specific kinds of information see visual.py

Last updated February 2016

by Trevor Arp
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import sleep
from os import getcwd as cwd

from scans import range_from_log

from new_colormaps import _viridis_data, _plasma_data

matplotlib.rcParams["keymap.fullscreen"] = ''

'''
Returns the viridis colormap,

for use before matplotlib is updated
'''
def get_viridis():
	return mcolors.ListedColormap(_viridis_data, name='Viridis')
#

'''
Returns the plasma colormap,

for use before matplotlib is updated
'''
def get_plasma():
	return mcolors.ListedColormap(_plasma_data, name='Plasma')
#

'''
DEPRICIATED in favor of viridis
Returns the color map for plotting
$cpt is the central point of the color transition (0 to 1 scale)
$width is related to the width of the color transition (0 to 1 scale)
'''
def generate_colormap(cpt=0.5, width=0.25):
	low_RGB = (249/255., 250/255., 255/255.)
	cl_RGB = (197/255., 215/255., 239/255.)
	clpt = np.max([0.05, cpt-width])
	center_RGB = (100/255., 170/255., 211/255.)
	ch_RGB = (30/255., 113/255., 180/255.)
	chpt = np.min([0.95, cpt+width])
	high_RGB =  (8/255., 48/255., 110/255.)
	cdict = {'red':((0.0, 0.0, low_RGB[0]),
					(clpt, cl_RGB[0], cl_RGB[0]),
					(cpt, center_RGB[0], center_RGB[0]),
					(chpt, ch_RGB[0], ch_RGB[0]),
					(1.0, high_RGB[0], 0.0)),


			'green': ((0.0, 0.0, low_RGB[1]),
					(clpt, cl_RGB[1], cl_RGB[1]),
					(cpt, center_RGB[1], center_RGB[1]),
					(chpt, ch_RGB[1], ch_RGB[1]),
					(1.0, high_RGB[1], 0.0)),

			'blue':  ((0.0, 0.0, low_RGB[2]),
					  (clpt, cl_RGB[2], cl_RGB[2]),
					  (cpt, center_RGB[2], center_RGB[2]),
					  (chpt, ch_RGB[2], ch_RGB[2]),
					  (1.0, high_RGB[2], 0.0)),
		  }
	return mcolors.LinearSegmentedColormap('GreenBlue', cdict)
# end generate_colormap

'''
Formats the plot axes in a standard format
$ax is the axes object for the plot, such as plt.gca()
'''
def format_plot_axes(ax, fntsize=14, tickfntsize=12):
	for i in ax.spines.itervalues():
		i.set_linewidth(2)
	ax.tick_params(width=2, labelsize=tickfntsize, direction='out')
	matplotlib.rcParams.update({'font.size': fntsize})
# end format_plot_axes

'''
Sets the x and y ticks for a data image based on the log file
$ax is the current axes
$img is the image object
$log is the log file

$xparam and $yparam are the paramters for the x and y axes if they are strings set the from the
log file, if they are a numpy array they are set from that array

$nticks is the number of ticks to use
$sigfigs is the number of significant figures to round to

$aspect if true will fix teh apsect ratio to the given value, usefull if the axes have different units
'''
def set_img_ticks(ax, img, log, xparam, yparam, nticks=5, sigfigs=2, aspect=None):
	if isinstance(xparam, str):
		xt = np.linspace(0, int(log['nx'])-1, nticks)
		xrng = range_from_log(xparam, log, log['nx'])
	elif isinstance(xparam, np.ndarray):
		xt = np.linspace(0, len(xparam)-1, nticks)
		xrng = xparam
	else:
		print 'Error set_img_ticks: X Parameter must be a string or an array, received: ' + str(xparam)
		return
	if isinstance(yparam, str):
		yt = np.linspace(0, int(log['ny'])-1, nticks)
		yrng = range_from_log(yparam, log, log['ny'])
	elif isinstance(yparam, np.ndarray):
		yt = np.linspace(0, len(yparam)-1, nticks)
		yrng = yparam
	else:
		print 'Error set_img_ticks: Y Parameter must be a string or an array, received: ' + str(yparam)
		return
	xl = xrng[xt.astype(int)]
	yl = yrng[yt.astype(int)]
	#print yl
	for i in range(len(xl)):
	    xl[i] = round(xl[i], sigfigs)
	for i in range(len(yl)):
	    yl[i] = round(yl[i], sigfigs)
	extent = (xl[0], xl[len(xt)-1], yl[len(yt)-1], yl[0])
	img.set_extent(extent)
	if aspect is not None:
		ax.set_aspect(float(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect))
	ax.set_xticks(xl)
	ax.set_yticks(yl)
	ax.set_xlim(xl[0], xl[len(xt)-1])
	ax.set_ylim(yl[len(yt)-1], yl[0])
# end set_img_ticks
