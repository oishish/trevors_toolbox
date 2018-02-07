'''
display.py

A module for general functions related to displaying information, for functions and classes related
to displaying specific kinds of information see visual.py

Last updated March 2017

by Trevor Arp
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import numpy as np
from time import sleep
from os import getcwd as cwd

from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift

from scans import range_from_log
from fitting import power_law, power_law_fit

#from new_colormaps import _viridis_data, _plasma_data

matplotlib.rcParams["keymap.fullscreen"] = ''

'''
A class to assist in the layout of figures, handeling the conversion from matplotlibs tricky
0 to 1 units to acutal physical units in inches.

Will also generate figures and axes of various kinds from specifications given in inches
'''
class figure_inches():
    '''
    Initilizer, takes the acutal size of the figure in inches
    '''
    def __init__(self, name, xinches, yinches):
        self.xinches = xinches
        self.yinches = yinches
        self.r = yinches/xinches
        self.name = name
        self.fig = plt.figure(self.name, figsize=(self.xinches, self.yinches), facecolor='w')
    # end init

    '''
    Returns a figures with the physical size given
    '''
    def get_fig(self):
        return self.fig
    # end make_figure

    '''
    Makes and returns axes with coordinates [left, bottom, width, height] in inches
    '''
    def make_axes(self, spec, zorder=1):
        plt.figure(self.name)
        return plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder)
    # make_axes

    '''
    Makes and returns two overlaid axes, with two y axes sharing the same x-axis with
    coordinates [left, bottom, width, height] in inches

    Note: the first axes returned (the left y-axis) is "on top" and provides the x-axis

    Left and right y-axes can be colored differently
    '''
    def make_dualy_axes(self, spec, color_left='k', color_right='k', zorder=1):
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axl = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.tick_params('y', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)

        axr = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', colors=color_right)
        return axl, axr
    # make_dualy_axes

    '''
    Makes and returns two overlaid axes, with two x axes sharing the same y-axis with
    coordinates [left, bottom, width, height] in inches

    Note: the first axes returned (the bottom x-axis) is "on top" and provides the y-axis

    Top and bottom x-axes can be colored differently
    '''
    def make_dualx_axes(self, spec, color_bottom='k', color_top='k', zorder=1):
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.tick_params('x', colors=color_bottom)
        axb.spines['bottom'].set_color(color_bottom)
        axb.spines['top'].set_color(color_top)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', colors=color_top)
        return axb, axt
    # make_dualx_axes

    '''
    Makes and returns four overlaid axes corresponding to each
    coordinates [left, bottom, width, height] in inches

    Note: the first axis returned (the left y-axis) is "on top"

    The return order if left, right, bottom, top axis
    '''
    def make_dualxy_axes(self, spec, color_bottom='k', color_top='k', color_left='k', color_right='k', zorder=1):
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axl = plt.axes([spec[0]+4, spec[0]+4, spec[0]+4, spec[0]+4], zorder=zorder+3)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.xaxis.set_visible(False)
        axl.tick_params('y', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)
        axl.spines['top'].set_color(color_top)
        axl.spines['bottom'].set_color(color_bottom)

        axr = plt.axes([spec[0]+3, spec[0]+3, spec[0]+3, spec[0]+3], zorder=zorder+2)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', colors=color_right)

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.yaxis.set_visible(False)
        axb.xaxis.tick_bottom()
        axb.xaxis.set_label_position("bottom")
        axb.tick_params('x', colors=color_bottom)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', colors=color_top)
        return axl, axr, axb, axt
    # make_dualx_axes
# end figure_inches

"""
Initilizes a figure with a specified size based on the number of rows and columns

One of the plots can be 3D, pass the coordinates to ax3D as a tuple
"""
def get_figure(fignum, rows=1, cols=1,
	fntsize=18,
	tickfntsize=16,
    basewidth=6.5,
    baseheight=5.0,
    margin=0.09,
    marginwidth=2.0,
    marginheight=1.0,
	fancyfonts=True,
	ax3D=None ):
	if fancyfonts:
		fancy_fonts()
		matplotlib.rcParams.update({'font.size': fntsize})
		matplotlib.rcParams.update({'axes.labelpad': 0})
	axes = []
	fig = plt.figure(fignum, figsize=(cols*basewidth+marginwidth, rows*baseheight+marginheight), facecolor='w')
	axspec = (rows, cols)
	if ax3D is not None:
		i3d = ax3D[0]
		j3d = ax3D[1]
	else:
		i3d = -1
		j3d = -1
	for i in range(rows):
		for j in range(cols):
			if i == i3d and j == j3d:
				ax = plt.subplot2grid(axspec,(i,j), projection='3d')
			else:
				ax = plt.subplot2grid(axspec,(i,j))
			format_plot_axes(ax, fntsize=fntsize, tickfntsize=tickfntsize)
			axes.append(ax)
	if (rows==1 and cols==1):
		margin = 0.07
		plt.subplots_adjust(
		left  = 1.43*margin, # the left side of the subplots of the figure
		right = 1.0-0.8*margin, # the right side of the subplots of the figure
		bottom = 1.5*margin, # the bottom of the subplots of the figure
		top = 1.0-1.5*margin, # the top of the subplots of the figure
		wspace = 2.7*margin, # the amount of width reserved for blank space between subplots
		hspace = 2.7*margin # the amount of height reserved for white space between subplots
		)
	elif (rows == 2 and cols==2):
		margin = 0.07
		plt.subplots_adjust(
		left  = margin, # the left side of the subplots of the figure
		right = 1.0-margin, # the right side of the subplots of the figure
		bottom = margin, # the bottom of the subplots of the figure
		top = 1.0-margin, # the top of the subplots of the figure
		wspace = 2.7*margin, # the amount of width reserved for blank space between subplots
		hspace = 2.7*margin # the amount of height reserved for white space between subplots
		)
	else:
		plt.subplots_adjust(
		left  = margin, # the left side of the subplots of the figure
		right = 1.0-0.75*margin, # the right side of the subplots of the figure
		bottom = 1.25*margin, # the bottom of the subplots of the figure
		top = 1.0-1.25*margin, # the top of the subplots of the figure
		wspace = 2*margin, # the amount of width reserved for blank space between subplots
		hspace = 2*margin # the amount of height reserved for white space between subplots
		)
	return axes, fig
# end get_figure()

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
Sets a plot to use a scale bar and hidden axis,
Always makes the scale bar in the lower right

Should be called in place of set_img_ticks, for 2D scans with hidden axis
$ax is the current axes
$img is the image object
$log is the log file

default paramters:
$length=2 is the length of the colorbar in data units
$units is the text to put after the length above the scale bar
$color is the color to use, white by default
$fontsize
'''
def scale_bar_plot(ax, img, log, length=2, units=r'$\mu m$', color='w', fontsize=16):
	lx = log['Fast Axis End'] - log['Fast Axis Start']
	xvals = np.linspace(-lx/2, lx/2, 5)
	ly = log['Slow Axis End'] - log['Slow Axis Start']
	yvals = np.linspace(-ly/2, ly/2, 5)
	set_img_ticks(ax, img, log, xvals, yvals, nticks=5)
	x0 = lx/2 - lx/15 - length
	y0 = ly/2 - ly/15 - length/10.0
	fnt = {'family':'STIXGeneral',
			'size':fontsize}
	ax.add_patch(mpatches.Rectangle((x0,y0), length, length/10.0, facecolor=color, edgecolor=color))
	ax.text(x0+length/2, y0-ly/30, str(length)+' '+units, color=color, fontsize=fontsize, horizontalalignment='center', fontdict=fnt)
	ax.axis('off')
# end scale_bar_plot

'''
Defines a standard format for "notebook" figures and basic visualizations
'''
def figure_format(fntsize=14, lw=1.0, labelpad=5):
	matplotlib.rc('font', **{'size':fntsize}) # Temporary Workaround
	# matplotlib.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':fntsize})
	# matplotlib.rc('text', usetex=True)
	# matplotlib.rcParams['text.latex.preamble'] = [
	#        r'\usepackage{helvet}',    # set the normal font here
	#        r'\usepackage{sansmathfonts}',  # load up the sansmath so that math -> helvet
	#        r'\usepackage{amsmath}'
	# ]
	matplotlib.rcParams.update({'axes.labelpad': labelpad})
	matplotlib.rcParams.update({'xtick.direction':'out'})
	matplotlib.rcParams.update({'ytick.direction':'out'})
	matplotlib.rcParams.update({'xtick.major.width':lw})
	matplotlib.rcParams.update({'ytick.major.width':lw})
	matplotlib.rcParams.update({'axes.linewidth':lw})
	matplotlib.rcParams.update({'image.interpolation':'bilinear'})
# figure_format

'''
Defines a standard format for paper figures and other production quality visualizations,
uses helvetical font and tex rendering
'''
def paper_figure_format(fntsize=15):
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
# end paper_figure_format

'''
Sets the x and y ticks for a data image based on the log file
$ax is the current axes
$img is the image object
$log is the log file

$xparam and $yparam are the paramters for the x and y axes if they are strings set the from the
log file, if they are a numpy array they are set from that array

$nticks is the number of ticks to use
$sigfigs is the number of significant figures to round to

$aspect if true will fix the apsect ratio to the given value, usefull if the axes have different units
'''
def set_img_ticks(ax, img, log, xparam, yparam, nticks=5, sigfigs=2, aspect=None):
	if isinstance(xparam, str):
		xt = np.linspace(0, int(log['nx'])-1, nticks)
		xrng = range_from_log(xparam, log, log['nx'])
	elif isinstance(xparam, np.ndarray):
		xt = np.linspace(0, len(xparam)-1, nticks)
		xrng = xparam
	else:
		print('Error set_img_ticks: X Parameter must be a string or an array, received: ' + str(xparam))
		return
	if isinstance(yparam, str):
		yt = np.linspace(0, int(log['ny'])-1, nticks)
		yrng = range_from_log(yparam, log, log['ny'])
	elif isinstance(yparam, np.ndarray):
		yt = np.linspace(0, len(yparam)-1, nticks)
		yrng = yparam
	else:
		print('Error set_img_ticks: Y Parameter must be a string or an array, received: ' + str(yparam))
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

'''
Takes a color map and returns a new colormap that only uses the part of it between minval and maxval
on a scale of 0 to 1

Taken From: http://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
'''
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# end truncate_colormap


'''
Returns a colormap, Normlization and ScalarMappable for given data

if color bounds not specified the min and max of the array are used, the mappable is initilized
to the data
'''
def colorscale_map(darray, cmin=None, cmax=None , mapname='viridis'):
	cmap = plt.get_cmap(mapname)
	if cmin is None:
		cmin = np.min(darray)
	if cmax is None:
		cmax = np.max(darray)
	cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
	scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
	scalarMap.set_array(darray)
	return cmap, cNorm, scalarMap
# end colorscale_map

'''
Returns a discrete colorscale to plot the data on

rngdict is a dictionary of ranges to show with a color scale with labels as keys for example,
rngdict={"<1/2":(0.0,0.45), "1/2":(0.45,0.55), ">1/2":(0.55,1.0)}, in increasing order

Returns: tick_locations, tick_labels, colorscale, normalization
'''
def discrete_colorscale(rngdict, basecolormap='viridis'):
    labels = []
    locations = []
    bounds = []
    bnds = []
    for k,v in rngdict.items():
        labels.append(k)
        locations.append((v[1]+v[0])/2.0)
        bnds.append(v[0])
        bnds.append(v[1])
    for i in bnds:
        if i not in bounds:
            bounds.append(i)
    bounds.sort()
    cmap = cm.get_cmap(basecolormap,len(rngdict))
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return locations, labels, cmap, norm
# end discrete_colorscale


'''
DEPRICAITED FUNCTIONS
'''

'''
Formats the plot axes in a standard format
$ax is the axes object for the plot, such as plt.gca()

DEPRICIATED in favor of figure_format()
'''
def format_plot_axes(ax, fntsize=16, tickfntsize=14):
	for i in ax.spines.values():
		i.set_linewidth(2)
	ax.tick_params(width=2, labelsize=tickfntsize, direction='out')
	matplotlib.rcParams.update({'font.size': fntsize})
# end format_plot_axes

'''
Set's the default font to the STIX font family

DEPRICIATED in favor of figure_format()
'''
def fancy_fonts():
	matplotlib.rcParams['mathtext.fontset'] = 'stix'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
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
Returns the viridis colormap,

DEPRICIATED since the new colormaps were added to matplotlib, still here for backwards compatibility
'''
def get_viridis():
	# return mcolors.ListedColormap(_viridis_data, name='Viridis')
	return plt.get_cmap('viridis')
#

'''
Returns the plasma colormap,

DEPRICIATED since the new colormaps were added to matplotlib, still here for backwards compatibility
'''
def get_plasma():
	return plt.get_cmap('plasma')
	# return mcolors.ListedColormap(_plasma_data, name='Plasma')
#
