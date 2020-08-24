'''
display.py

A module for general functions related to displaying information, for functions and classes related
to displaying specific kinds of information see visual.py

Last updated August 2020

by Trevor Arp
All Rights Reserved
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import numpy as np

from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift

from toolbox.fitting import power_law, power_law_fit

matplotlib.rcParams["keymap.fullscreen"] = ''

class figure_inches():
    '''
    A class to assist in the layout of figures, handling the conversion from matplotlibs
    0 to 1 units to actual physical units in inches. Will also generate figures and axes
    of various kinds from specifications given in inches.

    Args:
        name (str) : The name of the figure, can't have two figures with the same name in a script
        xinches (float) : The width of the figure in inches
        yinches (float) : The height of the figure in inches
    '''

    def __init__(self, name, xinches, yinches):
        self.xinches = xinches
        self.yinches = yinches
        self.r = yinches/xinches
        self.name = name
        self.fig = plt.figure(self.name, figsize=(self.xinches, self.yinches), facecolor='w')
    # end init

    def get_fig(self):
        '''
        Returns a figures object with the physical size given
        '''
        return self.fig
    # end make_figure

    def make_axes(self, spec, zorder=1):
        '''
        Makes and returns a matplotlib Axes object.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        plt.figure(self.name)
        return plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder)
    # make_axes

    def make_3daxes(self, spec, zorder=1):
        '''
        Makes and returns a matplotlib Axes object with a 3D projection

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        plt.figure(self.name)
        return plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder, projection='3d')
    # make_3daxes

    def make_dualy_axes(self, spec, color_left='k', color_right='k', zorder=1, lefthigher=True):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
            lefthigher (bool, optional) : If True (default) the left axis will be on top and provide the x-axis.
        '''
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        if lefthigher:
            zorderl = zorder + 1
            zorderr = zorder
        else:
            zorderl = zorder
            zorderr = zorder + 1
        #

        axl = plt.axes([spec[0]+1, spec[1]+1, spec[2]+1, spec[3]+1], zorder=zorderl)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.tick_params('y', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)

        axr = plt.axes([spec[0]+2, spec[1]+2, spec[2]+2, spec[3]+2], zorder=zorderr)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', colors=color_right)
        return axl, axr
    # make_dualy_axes

    def make_dualx_axes(self, spec, color_bottom='k', color_top='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note, the
        first axes returned (the bottom x-axis) is "on top" and provides the y-axis

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
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

    def make_dualxy_axes(self, spec, color_bottom='k', color_top='k', color_left='k', color_right='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note: the
        first axis returned (the left y-axis) is "on top"

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.

        Returns:
            axes : Returns the axes in the following order: left, right, bottom, top
        '''
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

def yaxis_right(ax):
    '''
    A simple function to move the y-axis of a matplotlib plot to the right.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
# end yaxis_right

def xaxis_top(ax):
    '''
    A simple function to move the x-axis of a matplotlib plot to the top.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
# end yaxis_right

def figure_format(fntsize=12, font=None, bilinear=True, labelpad=0):
    '''
    Defines a standard format for paper figures and other production quality visualizations,
    can use any font that is in the maplotlib font folder Lib/site-packages/matplotlib/mpl-data/fonts/tff,

    Args:
        fntsize (int, optional) : The fontsize
        font (int, optional) : The font, in None (default) uses Arial
        bilinear (bool, optional) : If true (default) will use 'bilinear' interpolation option in imshow
        labelpad (float, optional) : The default padding between the axes and axes labels.
    '''
    matplotlib.rcParams.update({'font.family':'sans-serif'})
    if font is not None:
        matplotlib.rcParams.update({'font.sans-serif':font})
    else:
        matplotlib.rcParams.update({'font.sans-serif':'Arial'})
    matplotlib.rcParams.update({'font.size':fntsize})
    matplotlib.rcParams.update({'axes.labelpad': labelpad})
    matplotlib.rcParams.update({'axes.titlepad': labelpad})
    matplotlib.rcParams.update({'xtick.direction':'out'})
    matplotlib.rcParams.update({'ytick.direction':'out'})
    matplotlib.rcParams.update({'xtick.major.width':1.0})
    matplotlib.rcParams.update({'ytick.major.width':1.0})
    matplotlib.rcParams.update({'axes.linewidth':1.0})
    if bilinear:
        matplotlib.rcParams.update({'image.interpolation':'bilinear'})
# end paper_figure_format

def change_axes_colors(ax, c):
    '''
    Changes the color of a given matplotlib Axes instance

    Note: for colorbars will need to use matplotlib.rcParams['axes.edgecolor'] = c before
    calling colorbarBase because whoever coded that clase bound the colorbar axes to that
    param value rather than having it function like a normal axis, which it is.

    Args:
        ax : The matplotlib axes object to manipulate
        c : The color, in matplotlib notation.
    '''
    ax.yaxis.label.set_color(c)
    ax.xaxis.label.set_color(c)
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
# end change_axes_colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    '''
    Takes a color map and returns a new colormap that only uses the part of it between minval and maxval
    on a scale of 0 to 1

    Taken From: http://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib

    Args:
        cmap : The matplotlib colormap object to truncate
        minval (float, optional) : The minimum colorvalue, on a scale of 0 to 1
        maxval (float, optional) : The maximum colorvalue, on a scale of 0 to 1
        n (int, optional) : The number of samples, if -1 (default) uses the sampling from cmap
    '''
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# end truncate_colormap

def colorscale_map(darray, mapname='viridis', cmin=None, cmax=None, centerzero=False, truncate=None):
    '''
    Generates a Colormap, Normalization and ScalarMappable objects for the given data.

    Args:
        darray : The data, If color bounds not specified the min and max of the array are used,
            the mappable is initialized to the data.
        mapname : The name (matplotlib conventions) of the colormap to use.
        cmin (float, optional) : The minimum value of the map
        cmax (float, optional) : The maximum value of the map
        centerzero (bool, optional) : If true will make zero the center value of the colorscale, use
            for diverging colorscales.
        truncate (tuple, optional) : If not None (default) will truncate the colormap with (min, max)
            colorvalues on a scale of 0 to 1.

    Returns:
        Tuple containing (cmap, norm, sm) where cmap is the Colormap, norm is the
        Normalization and sm is the ScalarMappable.
    '''
    cmap = plt.get_cmap(mapname)
    if truncate is not None:
        cmap = truncate_colormap(cmap, truncate[0], truncate[1])
    if cmin is None:
        cmin = np.min(darray)
    if cmax is None:
        cmax = np.max(darray)
    if centerzero:
        cmax = max(np.abs([cmin, cmax]))
        cmin =  -1.0*cmax
    cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap.set_array(darray)
    return cmap, cNorm, scalarMap
# end colorscale_map

def make_colorbar(ax, cmap, cnorm, orientation='vertical', ticks=None, ticklabels=None, color='k', alpha=None):
    '''
    Instantiates and returns a colorbar object for the given axes, with a few more options than
    instantiating directly

    Args:
        ax : The axes to make the colorbar on.
        cmap : The Colormap
        norm : The Normalization
        orientation (str, optional) : 'vertical' (default) or 'horizontal' orientation
        ticks (list, optional) : the locations of the ticks. If None will let matplotlib automatically set them.
        ticklabels (list, optional) : the labels of the ticks. If None will let matplotlib automatically set them.
        color (str, optional) : the color of the colorbar, default black.
        alpha (float, optional) : the transparency of the colorbar

    Returns:
        The matplotlib.ColorbarBase object.
    '''
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm, orientation=orientation, ticks=ticks, alpha=None)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    if color != 'k':
        matplotlib.rcParams['axes.edgecolor'] = color
        change_axes_colors(ax, color)
    return cb
# end make_colorbar

def discrete_colorscale(rngdict, basecolormap='viridis'):
    '''
    Creates a discrete colorscale.

    Args:
        rngdict (dict) : The ranges to show with a color scale with labels as keys for example,
            rngdict={"<1/2":(0.0,0.45), "1/2":(0.45,0.55), ">1/2":(0.55,1.0)}, in increasing order.
        basecolormap (str, optional) : the continuous colormap to base the discrete colormap on, default 'viridis'.

    Returns: tick_locations, tick_labels, colorscale, normalization
    '''
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

def fix_label_misalign(ax, offset=-0.0045, fixX=True):
	"""
	Fixes the bug where the positive and negative tick labels have an offset

	Displaces the positive labels by the offset paramter

	Fixes the labels on the x axis is fixX is true, the y axis labels otherwise
	"""
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

def zoom_in_center(ax, percentage):
	'''
	Zooms in on central $percentage area of a 2D figure
	'''
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

def center_2D_data(d, mode='wrap', cval=0.0):
	'''
	Centers the data in the middle of the map

	$d is a 2D arrawy of data to center

	$mode specifies what do do with the points outside the boundaries, passed to
	scipy.ndimage.interpolation.shift, default is wrap

	$fillvall if the mode is 'constant' fills out of bounds points with this value
	'''
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

def show_powerlaw_points(aximg, axplt, log, x, y, power, d, color=None, showerr=False):
	'''
	Plots a power law from a point on a 2D map, puts markers on the map
	'''
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

def _wavelength_to_rgb(wavelength, gamma=0.8):
	'''
	This converts a given wavelength of light to an approximate RGB color value. The wavelength must be given in nanometers in the range from 380 nm through 750 nm.

	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html
	'''
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

def wavelength_colormap(N=250):
	'''
	Creates a normalized colormap the converts a wavelength (in nm) to approximatly the color of the
	light with that wavelength
	'''
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
