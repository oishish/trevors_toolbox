'''
display.py

A module for general functions related to displaying information, for functions and classes related
to displaying specific kinds of information see visual.py

Last updated February 2020

by Trevor Arp
'''

import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import numpy as np

from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift

from toolbox.fitting import power_law, power_law_fit

matplotlib.rcParams["keymap.fullscreen"] = ''


"""
Fixes the bug where the positive and negative tick labels have an offset

Displaces the positive labels by the offset paramter

Fixes the labels on the x axis is fixX is true, the y axis labels otherwise
"""
def fix_label_misalign(ax, offset=-0.0045, fixX=True):
	if fixX:
		tks = ax.get_xticklabels()
		xticks = ax.get_xticks().tolist()
		for i in range(len(tks)):
		    if float(xticks[i]) >= 0.0:
		        tks[i].set_y(offset)
	else:
		tks = ax.get_yticklabels()
		yticks = ax.get_yticks().tolist()
		for i in range(len(tks)):
		    if float(yticks[i]) >= 0.0:
		        tks[i].set_x(offset)
# ebd fix_label_misalign

'''
Zooms in on central $percentage area of a 2D figure
'''
def zoom_in_center(ax, percentage):
    ymax, ymin = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    dy = np.abs(ymax - ymin)
    dx = np.abs(xmax - xmin)
    y0 = max(ax.get_ylim()) - dy/2
    x0 = max(ax.get_xlim()) - dx/2
    sp = np.sqrt(percentage)
    ax.set_ylim(y0+sp*dy/2, y0-sp*dy/2)
    ax.set_xlim(x0-sp*dx/2, x0+sp*dx/2)
# end zoom_in_center

'''
Centers the data in the middle of the map

$d is a 2D arrawy of data to center

$mode specifies what do do with the points outside the boundaries, passed to
scipy.ndimage.interpolation.shift, default is wrap

$fillvall if the mode is 'constant' fills out of bounds points with this value
'''
def center_2D_data(d, mode='wrap', cval=0.0):
	rows, cols = d.shape
	coords = center_of_mass(np.abs(d))
	dr = rows/2 - coords[0]
	dc = cols/2 - coords[1]
	if mode == 'constant':
		d = shift(d, (dr, dc), mode=mode, cval=0.0)
	else:
		d = shift(d, (dr, dc), mode=mode)
	return d
# end center_2D_data

'''
Plots a power law from a point on a 2D map, puts markers on the map
'''
def show_powerlaw_points(aximg, axplt, log, x, y, power, d, color=None, showerr=False):
	if color is None:
		color=['b','r','m','c']
	fx = log['Fast Axis']
	fc = (fx[1]-fx[0])/log['nx']
	sx = log['Slow Axis']
	sc = (sx[1]-sx[0])/log['ny']
	for i in range(len(x)):
		aximg.plot([fx[0]+fc*x[i]], [sx[0]+sc*y[i]], color[i]+'o')
		pc = d[y[i],x[i],:]
		if len(power.shape) == 1:
			pw = power[:]
		elif len(power.shape) == 2:
			pw = power[y[i],:]
		elif len(power.shape) == 3:
			pw = power[y[i], x[i], :]
		else:
			print("Error show_powerlaw_points: Invalid Power Data")
			return
		params, err = power_law_fit(pw-np.min(pw), pc)
		if showerr:
			lbl = r"$\gamma = $ "+ str(round(params[1],2)) + r' $\pm$' + str(round(err[1],2))
		else:
			lbl = r"$\gamma = $ "+ str(round(params[1],2))
		axplt.plot(pw, power_law(pw-np.min(pw), params[0], params[1], params[2]), color[i]+'-', lw=2 , label=lbl)
		axplt.plot(pw, pc, color[i]+'o', lw=2)
# end show_powerlaw_points

'''
Defines a standard format for paper figures and other production quality visualizations,
uses helvetical font and tex rendering
WARNING: With TeX rendering it may be unable to save .svg files
'''
def tex_figure_format(fntsize=15):
    matplotlib.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':fntsize})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{helvet}',    # set the normal font here
        r'\usepackage{sansmathfonts}',  # load up the sansmath so that math -> helvet
        r'\usepackage{amsmath}'
    ]
    matplotlib.rcParams.update({'axes.labelpad': 0})
    matplotlib.rcParams.update({'xtick.direction':'out'})
    matplotlib.rcParams.update({'ytick.direction':'out'})
    matplotlib.rcParams.update({'xtick.major.width':1.0})
    matplotlib.rcParams.update({'ytick.major.width':1.0})
    matplotlib.rcParams.update({'axes.linewidth':1.0})
    matplotlib.rcParams.update({'image.interpolation':'bilinear'})
# end paper_figure_format

'''
This converts a given wavelength of light to an approximate RGB color value. The wavelength must be given in nanometers in the range from 380 nm through 750 nm.

Based on code by Dan Bruton
http://www.physics.sfasu.edu/astro/color/spectra.html
'''
def _wavelength_to_rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    #
    return (R, G, B)
# end wavelength_to_rgb

'''
Creates a normalized colormap the converts a wavelength (in nm) to approximatly the color of the
light with that wavelength
'''
def wavelength_colormap(N=250):
    w = np.linspace(380, 750, N)
    colorvals = []
    for i in range(N):
        colorvals.append(_wavelength_to_rgb(w[i]))
    cmap = LinearSegmentedColormap.from_list('visiblespectrum', colorvals)
    cnorm = colors.Normalize(vmin=380, vmax=750)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    scalarMap.set_array(w)
    return cmap, cnorm, scalarMap
# end wavelength_colormap
