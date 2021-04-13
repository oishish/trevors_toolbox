'''
visual.py

A module for specific visualization routines

Last updated February 2016

by Trevor Arp
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from trevorarp.legacy import format_plot_axes

'''
A Class for displaying a data cube, where the user can switch between scans using a slider or using
the arrow keys.

$d is the data cube to be displayed.

$vrange is a tuple defining the range of the color scale, if None will autoscale

$trange is the title range is not None these values will be displayed for each scan

$title is the title of the graph, will be shown after $trange values if they exist

CubeFigure.ax gives the axes for manipulation
'''
class CubeFigure(object):
	def __init__(self, d, vrange=None, trange=None, title='', ndigits=2, cmap='viridis'):
		self.d = d

		rows, cols, N = self.d.shape
		self.N = N
		self.ndigits = ndigits

		self.fig = plt.figure()
		self.ax = self.fig.gca()

		self.ix = 1
		if vrange is not None:
			self.img = self.ax.imshow(self.d[:,:,self.ix-1], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
			self.autoZ = False
		else:
			self.img = self.ax.imshow(self.d[:,:,self.ix-1], cmap=cmap)
			self.autoZ = True
		self.cbar = self.fig.colorbar(self.img)

		self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
		self.slider = Slider(self.sliderax, 'Scan', 1, N, valinit=1)
		self.slider.on_changed(self.update_val)
		self.slider.drawon = False

		self.slider.valtext.set_text('{}'.format(int(self.ix))+'/'+str(self.N))

		self.title = title
		self.trange = trange
		if self.trange is not None:
			self.ax.set_title(str(round(self.trange[self.ix-1], self.ndigits)) + ' ' + self.title)
		else:
			self.ax.set_title(self.title)

		self.fig.canvas.mpl_connect('key_press_event', self.onKey)
		self.fig.show()
	# end init

	def onKey(self, event):
		k = event.key
		if (k == 'right' or k == 'up') and self.ix < self.N:
			val = self.ix + 1
			self.slider.set_val(val)
			self.update_val(val)
		elif (k == 'left' or k == 'down') and self.ix > 1:
			val = self.ix - 1
			self.slider.set_val(val)
			self.update_val(val)
		#
	# end onKey

	def update_val(self, value):
		self.ix = int(value)
		self.img.set_data(self.d[:,:,self.ix-1])
		if self.autoZ:
			self.img.autoscale()
		self.slider.valtext.set_text('{}'.format(int(self.ix))+'/'+str(self.N))
		if self.trange is not None:
			self.ax.set_title(str(round(self.trange[self.ix-1], self.ndigits)) + ' ' + self.title)
		self.update()
	# end update_val

	def update(self):
		self.fig.canvas.draw()
	# end update
# end CubeFigure

'''
A Class for dispalying the fit to a data cube and showing line cuts for a given point
'''
class Cube_Point_Display():
	'''
	Constructor
	$rn standard run number
	$d the data cube to plot
	$zdata the data in the z direction
	$fmap the map of the fit parameter to display
	$fit is the cube of the fit with fit parameters as a function of (x,y)
	$fitfunc is the function to fit to
	'''
	def __init__(self, rn, d, zdata, fmap, fit, fitfunc,
		xlabel='', ylabel='', zlabel='', vlabel='', title='', figtitle='', nargs=None
		):
		self.rn = rn
		self.d = d
		self.p = zdata
		self.fmap = fmap
		self.fit = fit
		self.f = fitfunc

		self.rows, self.cols, self.N = d.shape

		self.x = self.cols//2
		self.y = self.rows//2
		x = self.x
		y = self.y

		ri, cj, M = np.shape(self.fit)

		if nargs is None:
			self.nargs = M//2
		else:
			self.nargs = nargs

		self.fig = plt.figure(rn+'_'+figtitle, figsize=(14,6))

		# Set up the image of the fit parameter
		self.axf = plt.subplot(1,2,1)
		self.img = self.axf.imshow(self.fmap, cmap='viridis')
		self.axf.set_xlim(0, self.cols)
		self.axf.set_ylim(self.rows,0)
		self.cbar = self.fig.colorbar(self.img)

		self.pt = self.axf.plot(x,y, 'ko')
		self.pt = self.pt[0]
		self.axf.set_xlabel(xlabel)
		self.axf.set_ylabel(ylabel)
		self.axf.set_title(title)
		format_plot_axes(self.axf)

		# Set up the linecut image
		self.axc = plt.subplot(1,2,2)
		pc = self.d[y,x,:]

		if len(self.p.shape) > 1:
			pdata = self.p[x,:]
		else:
			pdata = self.p

		self.fline = self.axc.plot(pdata, self.f(pdata, *self.fit[y,x,0:self.nargs]), 'b-', lw=2)
		self.dpts = self.axc.plot(pdata, pc, 'bo', lw=2)
		self.axc.set_xlabel(zlabel)
		self.axc.set_ylabel(vlabel)
		self.axc.set_title(self.get_title())
		format_plot_axes(self.axc)
		# self.fig.tight_layout()

		# Set up the events
		fnc = lambda x : self.onClick(x)
		self.fig.canvas.mpl_connect('button_press_event', fnc)

		self.fig.show()
	# end init

	'''
	Handels the click event
	'''
	def onClick(self, event):
		if isinstance(event.xdata, (int, float)) and isinstance(event.ydata, (int, float)):
			self.x = int(event.xdata)
			self.y = int(event.ydata)
			self.updateLines()
	# end onClick

	'''
	Generates the title of the line cut axis, override for better functionality
	'''
	def get_title(self):
		return ""
	#

	'''
	Updates the axes
	'''
	def updateLines(self):
		x = self.x
		y = self.y
		pc = self.d[y,x,:]
		if len(self.p.shape) > 1:
			pdata = self.p[y,:]
		else:
			pdata = self.p
		self.fline[0].set_xdata(pdata)
		self.fline[0].set_ydata(self.f(pdata, *self.fit[y,x,0:self.nargs]))
		self.dpts[0].set_xdata(pdata)
		self.dpts[0].set_ydata(pc)
		self.axc.relim()
		self.axc.autoscale()
		self.axc.set_title(self.get_title())

		self.pt.set_xdata(x)
		self.pt.set_ydata(y)

		self.fig.canvas.draw()
	# end updateLines
# end Cube_Linecut_Display
