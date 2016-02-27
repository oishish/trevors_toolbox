'''
visual.py

A module for specific visualization routines

Last updated February 2016

by Trevor Arp
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from display import get_viridis, format_plot_axes

from fitting import power_law as fitting_power_law
import process
import postprocess
from utils import find_run

'''
A Class for displaying a data cube, where the user can switch between scans using a slider or using
the arrow keys.

$d is the data cube to be displayed.

CubeFigure.ax gives the axes for manipulation
'''
class CubeFigure(object):
	def __init__(self, d):
		self.d = d

		rows, cols, N = self.d.shape
		self.N = N

		self.fig = plt.figure()
		self.ax = self.fig.gca()

		self.ix = 1
		self.img = self.ax.imshow(self.d[:,:,self.ix-1], cmap=get_viridis())
		self.cbar = self.fig.colorbar(self.img)

		self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
		self.slider = Slider(self.sliderax, 'Scan', 1, N, valinit=1)
		self.slider.on_changed(self.update_val)
		self.slider.drawon = False

		self.slider.valtext.set_text('{}'.format(int(self.ix))+'/'+str(self.N))

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
		self.img.autoscale()
		self.slider.valtext.set_text('{}'.format(int(self.ix))+'/'+str(self.N))
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
	def __init__(self, rn, d, power, fmap, fit, fitfunc,
		xlabel='', ylabel='', zlabel='', vlabel='', title='', figtitle=''
		):
		self.rn = rn
		self.d = d
		self.p = power
		self.fmap = fmap
		self.fit = fit
		self.f = fitfunc

		self.rows, self.cols, self.N = d.shape

		self.x = self.cols/2
		self.y = self.rows/2
		x = self.x
		y = self.y

		self.fig = plt.figure(rn+'_'+figtitle, figsize=(14,6))

		# Set up the image of the fit parameter
		self.axf = plt.subplot(1,2,1)
		self.img = self.axf.imshow(self.fmap, cmap=get_viridis())
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
		pc = np.abs(self.d[y,x,:])
		pdata = self.p[x,:]
		self.fline = self.axc.plot(pdata, self.f(pdata, self.fit[y,x,0], self.fit[y,x,1]), 'b-', lw=2)
		self.dpts = self.axc.plot(pdata, pc, 'bo', lw=2)
		self.axc.set_xlabel(zlabel)
		self.axc.set_ylabel(vlabel)
		self.axc.set_title(self.get_title())
		format_plot_axes(self.axc)
		self.fig.tight_layout()
		self.fig.show()

		# Set up the events
		self.fig.canvas.mpl_connect('button_press_event', self.onClick)
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
		pc = np.abs(self.d[y,x,:])
		pdata = self.p[x,:]
		self.fline[0].set_xdata(pdata)
		self.fline[0].set_ydata(self.f(pdata, self.fit[y,x,0], self.fit[y,x,1]))
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

class Power_PCI_Cube_Point_Display(Cube_Point_Display):
	def __init__(self, run):
		rn = run.log['Run Number']
		power, drR, d, fit_drR, fit_pci = process.Space_Power_Cube(run, savefile=find_run(rn))
		gamma, gamma_err, rchi2 = postprocess.filter_power_cube(d, power, fit_pci, max_chi=0.2)
		Cube_Point_Display.__init__(self, rn, d, power, gamma, fit_pci, fitting_power_law,
			xlabel='Microns',
			ylabel='Microns',
			zlabel='Power (mW)',
			vlabel='|I| (nA)',
			title=rn+' Photocurrent '+r'$\gamma$',
			figtitle='pci'
			)
	# end init

	def get_title(self):
		return r"$\gamma = $ "+ str(round(self.fit[self.y,self.x,1],3)) + '$\pm$' + str(round(self.fit[self.y,self.x,3],2))
# end Power_PCI_Cube_Point_Display

class Power_RFI_Cube_Point_Display(Cube_Point_Display):
	def __init__(self, run):
		rn = run.log['Run Number']
		power, drR, d, fit_drR, fit_pci = process.Space_Power_Cube(run, savefile=find_run(rn))
		Cube_Point_Display.__init__(self, rn, drR, power, fit_drR[:,:,1], fit_drR, fitting_power_law,
			xlabel='Microns',
			ylabel='Microns',
			zlabel='Power (mW)',
			vlabel=r'$\Delta R/R$',
			title=rn+' Reflection '+r'$\gamma$',
			figtitle='rfi'
			)
	# end init

	def get_title(self):
		return r"$\gamma = $ "+ str(round(self.fit[self.y,self.x,1],3)) + '$\pm$' + str(round(self.fit[self.y,self.x,3],2))
# end Power_PCI_Cube_Point_Display
