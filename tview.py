'''

tview.py

Trevor's VIEWing Software

An event based viewer for data

DEPRICATED

'''

import numpy as np
import scipy as sp
import threading
import tkinter as tk
import tkinter.filedialog
import tkinter.filedialog
import sys
import os.path

import dimage
from display import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors



###################################################
# tview standalone app
###################################################

initial_directory = 'E:\Data'
save_directory = 'E:\Trevor'
default_color_cpt = 0.5
default_color_width = 0.25

class tview_app_toolbar(NavigationToolbar2TkAgg):
	def __init__(self,canvas_,parent_, viewer):
		self.viewer = viewer
		self.toolitems = (
		('Home', 'Reset original view', 'home', 'home'),
		#('Back', 'Back to  previous view', 'back', 'back'),
		#('Forward', 'Forward to next view', 'forward', 'forward'),
		(None, None, None, None),
		#('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
		('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
		(None, None, None, None),
		('Fiddle', 'Change colorscale', 'hand', 'fiddle'),
		('Lineplots', 'Toggle line cuts', 'subplots', 'toggle_plots'),
		('Save', 'Save the figure', 'filesave', 'save_figure'),
		)
		NavigationToolbar2TkAgg.__init__(self,canvas_,parent_)
	# end init

	def toggle_plots(self):
		self.viewer.toggle_line_plots()
	#

	def save_figure(self):
		self.viewer.save_figs()
	#

	def fiddle(self):
		if self._active == 'ZOOM':
			self.zoom()
		if self._active != 'FIDDLE':
			self._active = 'FIDDLE'
			self.mode = 'fiddle'
			self.set_message(self.mode)
		else:
			self._active = None
			self.mode = ''
			self.set_message(self.mode)
		#
	# end fiddle
# end tview_app_toolbar

class tview_app(threading.Thread):
	def __init__(self, data,
		file='',
		title='',
		xlabel='',
		ylabel='',
		xrange=(0,0),
		yrange=(0,0)
		):

		# Load the data and set number of images
		self.data = np.flipud(data)#data
		self.shape = np.shape(self.data)
		if len(self.shape) == 2:
			self.num_img = 1
			self.current = self.data
			self.current_ix = 0
		elif len(self.shape) == 3:
			self.current_ix = 0
			self.num_img = self.shape[2]
			self.current = self.data[:,:,self.current_ix]
		else:
			print("Error tview_app.py : Unrecognized data type")
			raise ValueError

		self.fname = file
		self.running = True
		self.screen = tk.Tk()
		self.screen.protocol("WM_DELETE_WINDOW", self.stop)

		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xrange = xrange
		self.yrange = yrange
		self.nticks = 7

		if title == '':
			temp = file.split('\\')
			self.title = str(temp[len(temp)-1])
		else:
			self.title = title
		self.screen.wm_title(self.title)

		self.cpt = default_color_cpt
		self.cwidth = default_color_width
		self.figsize = 8

		self.first_click_drag = True
		self.first_click = True
		self.disp_cuts = False

		self.x = self.shape[1]/2
		self.y = self.shape[0]/2

		# start the thread
		threading.Thread.__init__(self)
		self.start()
	# end init

	# Sets up then runs the program
	def run(self):
		# Initilize the image
		self.fig1 = plt.figure(figsize=(self.figsize, self.figsize))

		self.aspect_ratio = 1.0
		if self.xrange[1] != self.xrange[0] and self.yrange[1] != self.yrange[0]:
			ar = np.abs(1.0*(self.yrange[1]-self.yrange[0])/(self.xrange[1]-self.xrange[0]))
			if ar != self.aspect_ratio:
				self.aspect_ratio = ar
			#
		#

		# Main image
		#self.ax = plt.subplot2grid((2,3),(0,0), colspan=2, rowspan=2)
		self.ax = plt.gca()
		self.cmap = generate_colormap(cpt=self.cpt, width=self.cwidth)
		self.im = plt.imshow(self.current, cmap=self.cmap, aspect=self.aspect_ratio)
		divider = make_axes_locatable(self.ax)
		self.cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(cax=self.cax)
		#plt.colorbar()

		# Guides
		self.xguide = self.ax.plot([self.x, self.x], [0, self.shape[0]], 'k',lw=2)
		self.yguide = self.ax.plot([0, self.shape[1]], [self.y, self.y],'k',lw=2)
		self.xguide = self.xguide[0]
		self.yguide = self.yguide[0]
		self.ax.set_xlim(0, self.shape[1])
		self.ax.set_ylim(0, self.shape[0])
		self.xguide.set_visible(False)
		self.yguide.set_visible(False)



		# Set title, labels
		self.ax.set_title(self.title)
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		if self.xrange[0] != self.xrange[1]:
			self.ax.set_xticks(np.linspace(0, self.shape[1], self.nticks))
			xlb = np.linspace(self.xrange[0], self.xrange[1], self.nticks)
			xlb = np.around(xlb, decimals=2)
			self.ax.set_xticklabels(xlb)

		if self.yrange[0] != self.yrange[1]:
			self.ax.set_yticks(np.linspace(0, self.shape[0], self.nticks))
			ylb = np.linspace(self.yrange[1], self.yrange[0], self.nticks)
			ylb = np.around(ylb, decimals=2)
			self.ax.set_yticklabels(ylb)

		figures_miniFRAME = tk.Frame(self.screen)
		self.canvas = FigureCanvasTkAgg(self.fig1, master=figures_miniFRAME)
		self.tkwidget = self.canvas.get_tk_widget()
		self.tkwidget.grid(row=0, column=0) #.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

		# Line Cut subplots

		self.fig2 = plt.figure(2, figsize=(0.5*self.figsize, self.figsize))

		self.axx = plt.subplot(2,1,1)
		self.xplt = self.axx.plot(self.current[:,self.x])
		#self.axx.set_title('xplt')

		self.axy = plt.subplot(2,1,2)
		self.yplt = self.axy.plot(self.current[self.y,:])
		#self.axy.set_title('yplt')

		if self.xrange[0] != self.xrange[1]:
			self.axy.set_xticks(np.linspace(0, self.shape[1], self.nticks-1))
			xlb = np.linspace(self.xrange[0], self.xrange[1], self.nticks-1)
			xlb = np.around(xlb, decimals=2)
			self.axy.set_xticklabels(xlb)

		if self.yrange[0] != self.yrange[1]:
			self.axx.set_xticks(np.linspace(0, self.shape[0], self.nticks-1))
			ylb = np.linspace(self.yrange[1], self.yrange[0], self.nticks-1)
			ylb = np.around(ylb, decimals=2)
			self.axx.set_xticklabels(ylb)

		self.canvasxy = FigureCanvasTkAgg(self.fig2, master=figures_miniFRAME)
		self.tkwidgetxy = self.canvasxy.get_tk_widget()
		self.tkwidgetxy.grid(row=0, column=1) #.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
		self.tkwidgetxy.grid_remove()
		plt.tight_layout()

		figures_miniFRAME.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

		# Toolbar
		toolbar_miniFRAME = tk.Frame(self.screen)
		toolbar_miniFRAME.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
		self.toolbar = tview_app_toolbar(self.canvas, toolbar_miniFRAME, self)
		self.toolbar.update()
		self.toolbar.pack(side=tk.LEFT, expand=1)
		self.num_images_TEXT = tk.StringVar()
		self.num_images_TEXT.set("1/" + str(self.num_img))
		tk.Label(toolbar_miniFRAME, textvariable=self.num_images_TEXT, width=6).pack(side=tk.RIGHT)

		self.width = self.tkwidget.winfo_reqwidth()
		self.height = self.tkwidget.winfo_reqheight()

		# Bind Events
		#self.screen.bind('<ButtonPress-1>', self.on_click)
		self.screen.bind('<B1-Motion>', self.on_click_drag)
		self.screen.bind_all('<MouseWheel>', self.on_scroll)

		up_call = lambda e : self.on_arrow_press(0,1)
		down_call = lambda e : self.on_arrow_press(0,-1)
		left_call = lambda e : self.on_arrow_press(-1,0)
		right_call = lambda e : self.on_arrow_press(1,0)
		self.screen.bind('<Up>', up_call)
		self.screen.bind('<Down>', down_call)
		self.screen.bind('<Left>', left_call)
		self.screen.bind('<Right>', right_call)

		self.ax.set_picker(5)
		self.fig1.canvas.mpl_connect('pick_event', self.on_pick)

		self.screen.mainloop() # Must be last line
	# end run

	# Updates plot after press of an arrow key
	def on_arrow_press(self, dx, dy):
		if self.disp_cuts:
			self.x = max(min(self.x+dx, self.shape[1]-1),0)
			self.y = max(min(self.y+dy, self.shape[0]-1),0)
			self.update_xy()
		#
	#

	# updates the guides and line plots
	def update_xy(self):
		if self.disp_cuts:
			self.xguide.set_xdata([self.x, self.x])
			self.yguide.set_ydata([self.y, self.y])
			self.xplt[0].set_ydata(self.current[:,self.x])
			self.yplt[0].set_ydata(self.current[self.y,:])
			self.axx.relim()
			self.axy.relim()
			self.axx.autoscale_view()
			self.axy.autoscale_view()
		self.im.autoscale()
		self.fig1.canvas.draw()
		self.fig2.canvas.draw()
		#
	#


	# Stops the thread
	def stop(self):
		self.running = False
		self.screen.destroy()
		sys.exit()
	# end stop

	# Toggles displaying the line plots
	def toggle_line_plots(self):
		if self.disp_cuts:
			self.disp_cuts = False
			self.xguide.set_visible(False)
			self.yguide.set_visible(False)
			self.tkwidgetxy.grid_remove()
			self.fig1.canvas.draw()
			#print "Awww"
		else:
			self.disp_cuts = True
			self.xguide.set_visible(True)
			self.yguide.set_visible(True)
			self.tkwidgetxy.grid()
			self.fig1.canvas.draw()
			#print "Yay!"
		#
	#

	def save_figs(self):
		f = tkinter.filedialog.asksaveasfilename(initialdir=save_directory, initialfile=self.title)
		print(f)
	#

	def on_scroll(self, event):
		if self.num_img > 1:
			if event.delta > 0:
				self.current_ix = np.min([self.current_ix+1, self.num_img-1])
			else:
				self.current_ix = np.max([self.current_ix-1, 0])
			self.current = self.data[:,:,self.current_ix]
			self.num_images_TEXT.set(str(self.current_ix+1) + "/" + str(self.num_img))
			self.cpt = default_color_cpt
			self.cwidth = default_color_width
			self.cmap = generate_colormap(cpt=self.cpt, width=self.cwidth)
			self.im.set_cmap(self.cmap)
			self.im.set_data(self.current)
			self.update_xy()
		#
	# end on_scroll

	# Click-drag event
	def on_click_drag(self, event):
		if self.first_click_drag:
			self.last_drag_x = event.x
			self.last_drag_y = event.y
			self.first_click_drag = False
			return

		if self.toolbar._active == 'FIDDLE':
			inc = 0.01 # increment
			if (event.y - self.last_drag_y) > 0 and (self.cpt + self.cwidth + 5*inc) < 1:
				self.cpt = self.cpt + inc
			elif (event.y - self.last_drag_y) < 0 and (self.cpt - self.cwidth - 5*inc) > 0:
				self.cpt = self.cpt - inc
			if (event.x - self.last_drag_x) > 0 and self.cwidth + inc < 0.5:
				self.cwidth = self.cwidth + inc
			elif (event.x - self.last_drag_x) < 0 and self.cwidth - inc > 0.05:
				self.cwidth = self.cwidth - inc
			self.cmap = generate_colormap(cpt=self.cpt, width=self.cwidth)
			self.im.set_cmap(self.cmap)
			self.fig1.canvas.draw()
		self.last_drag_x = event.x
		self.last_drag_y = event.y
	# end on_click_drag

	# Updates the plots with the current position
	def on_pick(self, event):
		self.x = int(event.mouseevent.xdata)
		self.y = int(event.mouseevent.ydata)
		self.update_xy()
	#
# end tview_app

if __name__ == '__main__':
	if len(sys.argv) == 1:
		rt = tk.Tk()
		rt.withdraw()
		f = tkinter.filedialog.askopenfilename(initialdir=initial_directory, parent=rt)
		rt.destroy()
	else:
		f=sys.argv[1]
	if os.path.isfile(f):
		fnm, ext = os.path.splitext(f)
		if ext == '.npy':
			d = np.load(f)
		else:
			d = np.loadtxt(f)
		tview_app(d, file=f)
	else:
		print("Error in tview_app.py : cannot open file " + str(f))
