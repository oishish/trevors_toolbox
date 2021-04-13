'''
A place for things that interface with the older versions of the DAQ software,
and older versions of my code more generally.
'''
import numpy as np
from os.path import exists, join

from gaborlab.mpdpm import Run
from gaborlab.utilities import find_run, find_savefile
import standards as st

import matplotlib

from trevorarp.visual import Cube_Point_Display
from trevorarp.math import power_law

class LegacyRun(Run):
    '''
    A version of gaborlab.mpmdpm.Run that has been adapted to load legacy runs from before the
    switch to hyperDAQ in mid-2017. May behave unpredictably when used on more recent runs.

    Attributes:
        log : A dictionary of all the values in the log file
        axes : A list of numpy arrays giving all the axes values in order, i.e. [slow (i.e. Y), fast (i.e. X), cube, ...]
        units : A list of the empty strings since axes units are not explictly retained in older log files.
        labels : A list of the empty strings since axes labels are not retained the same way in older log files.
        data : A dictionary of the data images, where the keys are the image extensions and the values are arrays of the data
        data_units : A dictionary of the units (as strings) of all the axes, same keys as data
        shape : the shape of the data images, similar to numpy.shape

    Args:
        run_num (str) : The run number in standard hyperDAQ form, i.e. "SYSTEM_YEAR_MONTH_DAY_NUMBER"
        axis_labels (list) : Is a list specifyinf what labels ['Fast', 'Slow' and 'Cube (if applicable)'] have in the log file since they
            are not explicitly inlcuded in the log file in the older versions of the data. Will calculate axes from the 'Start' and 'End' entried for each
        **kwargs : The same optional arguments as mpdpm.Run
    '''
    def __init__(self, run_num, axis_labels, autosave=True, usecached=True, overwrite=False, calibrate=None, calibration_units=None, stabilize=True, preprocess=None, customdir=None):
        self.run_number = run_num
        # First find the file, if it exists
        if exists(run_num + '_log.log'):
            path = ''
        elif customdir is not None and find_run(run_num, directory=customdir) is not None:
            path = find_run(run_num, directory=customdir)
        elif find_run(run_num) is not None:
            path = find_run(run_num)
        else:
            print('Error mpdpm.Run : Could not open run :' + str(run_num))
            raise IOError
        #

        '''
        If it has a savefile in the processed directory, load the data and log from it
        '''
        if not overwrite and usecached:
            savefile = find_savefile(self.run_number, directory=customdir)
            if exists(join(savefile, self.run_number+"_run.npz")):
                r = self._load(savefile)
                if r == 0:
                    return
                else:
                    print("Warning: could not load savefile: " + join(savefile, self.run_number+"_run.npz") + " loading from raw data")
        #

        # Load in the log file
        file_path = join(path, run_num)
        with open(file_path + '_log.log', 'r') as fl:
            lg = fl.readlines()
        self.log = {}
        for line in lg:
            s = line.split(':')
            if len(s) == 2:
                k = s[0]
                v = s[1]
                try:
                    if '_' in v:
                        self.log[k] = str(v)
                    else:
                        self.log[k] = float(v)
                except ValueError:
                    self.log[k] = str(v).strip()
            elif len(s) > 2:
                k = s[0]
                v = s[1:len(s)]
                if k == 'Start Time':
                    self.log[k] = str(line.split('e:')[1])
                else:
                    self.log[k] = str(v).strip()
        # Define some standard names for the log file
        self.log['Fast Axis'] = (self.log['Fast Axis Start'], self.log['Fast Axis End'])
        self.log['Slow Axis'] = (self.log['Slow Axis Start'], self.log['Slow Axis End'])
        for output in st.card_ouput_entries:
            self.log[output] = (self.log[output+' Start'], self.log[output+' End'])
        #

        #load the data files
        self.data = {}
        types = self.log['Data Files']
        types = types.split(',')
        for s in types:
            if exists(file_path + '_' + s +'.dat'):
                self.data[s] = np.loadtxt(file_path + '_' + s +'.dat')
            elif exists(file_path + '_' + s +'.npy'):
                self.data[s] = np.load(file_path + '_' + s +'.npy')
            else:
                raise IOError("Error in mpmpm.Run: Cannot find data file for filetype: " + str(s))
        #

        # Define the shape
        self.shape = None
        for k,v in self.data.items():
            s = np.shape(v)
            if self.shape is None:
                self.shape = s
            else:
                if s != self.shape:
                    raise IOError("Error in mpmpm.Run: All data images must have the same shape")
            #
        # Pre-process if needed
        if preprocess is not None:
            self.process(preprocess)
        #

        # Calibrate the data images as specified in calibration specification
        if calibrate is None:
            calib = st.standard_image_calibration
        else:
            calib = calibrate
        for k,v in self.data.items():
            if calib[k] is not None:
                self.data[k] = calib[k](v, self.log)
        #

        # Assign units to the images
        self.data_units = {}
        if calibrate is None:
            calib_units = st.standard_image_calibration_units
        else:
            calib_units = calibration_units
        for k,v in self.data.items():
            if calib[k] is not None:
                self.data_units[k] = calib_units[k]
            else:
                self.data_units[k] = None
        #

        # Assemble the axes
        self.axes = []
        self.units = []
        self.labels = []
        if len(axis_labels) == 3 and len(self.shape) == 3:
            axlbls = ["Slow", "Fast", "Cube"]
        elif len(axis_labels) == 2 and len(self.shape) == 2:
            axlbls = ["Slow", "Fast"]
        else:
            raise ValueError("Invalid Number of Dimensions in axis_labels")
        #

        for i in range(len(axlbls)):
            self.units.append('') # Unfortunatly, older log files didn't keep track of these
            self.labels.append('') # Kept for compatability
            if axis_labels[i] == 'Angle': # The only measured axis in older versions was Power, i.e. 'Angle'
                img = 'pow'
                if axlbls[i] == "Cube":
                    ind = [0,1,2]
                else:
                    ind = [0,1]
                ind.remove(i)
                self.axes.append(np.mean(self.data[img], axis=tuple(ind))) # Average along the other axes
            else: # If it is a sampled axis
                # Before 2020 updates, sampling is always numpy.linspace
                self.axes.append(np.linspace(self.log[axis_labels[i] +' Start'], self.log[axis_labels[i] +' End'], self.shape[i]))
        #

        # stabilize the images if needed
        if stabilize:
            self._stabilize()
        #

        # Save the processed file
        if autosave:
            self.save(directory=customdir)
        #
    # end __init__

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
DEPRICATED in favor of figure_format()

Formats the plot axes in a standard format
$ax is the axes object for the plot, such as plt.gca()
'''
def format_plot_axes(ax, fntsize=16, tickfntsize=14):
	for i in ax.spines.values():
		i.set_linewidth(2)
	ax.tick_params(width=2, labelsize=tickfntsize, direction='out')
	matplotlib.rcParams.update({'font.size': fntsize})
# end format_plot_axes

'''
For a power data cube, filters the points based on the fit

$d is the raw data and $power is the power parameter

$fit is the calculated power law fit

kwargs:

$fill is the gamma value to fill the points that are filtered out, default (None) fills with np.nan

Returns filtered values of $gamma and $Amplitude
'''
def filter_power_cube(d, power, fit, fill=None, frac=1.0):
    rows, cols, N = d.shape
    if fill is None:
        fill = np.nan
    gamma = fit[:,:,1]
    params = fit[:,:,0:3]
    perr = fit[:,:,3:6]
    for i in range(rows):
        for j in range(cols):
            for k in range(2):
                if np.abs(perr[i,j,k]) > frac*np.abs(params[i,j,k]):
                    gamma[i,j] = 0.0
    return gamma
# end filter_power_cube

'''
For a power data cube, filters the points based on the signal amplitude

$d is the raw data and $power is the power parameter

$fit is the calculated power law fit

kwargs:

$fill is the gamma value to fill the points that are filtered out, default (None) fills with np.nan

$Amp is the minimum absolute value of the amplitude of the signal to be considered

Returns filtered values of $gamma
'''
def filter_power_cube_amplitude(d, power, fit, fill=None, Amp=1.0):
    rows, cols, N = d.shape
    if fill is None:
        fill = np.nan
    gamma = fit[:,:,1]
    params = fit[:,:,0:3]
    for i in range(rows):
        for j in range(cols):
            if np.abs(params[i,j,0]) < Amp:
                gamma[i,j] = 0.0
    return gamma
# end filter_power_cube


'''
Filters a Space-Delay cube, similar to filter_power_cube

$t is the time delay, and fit is the fit to a symmetric exponential

kwargs:

$fill is the tau value to fill the points that are filtered out, default (None) fills with np.nan

$min_A is the minimum amplitude of the fit function, points less that this are filtered out

$max_tau is the maximum acceptable value of tau, points above this are filtered out

$max_terr is the maximum acceptable error in tau, points above this are filtered out

returns filtered tau values

'''
def filter_delay_cube(t, fit, fill=None, min_A=0.1, max_tau=100, max_terr=5.0):
    if fill is None:
        fill = np.nan
    rows, cols, N = np.shape(fit)
    tau = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.abs(fit[i,j,1])<min_A or np.abs(fit[i,j,6])>max_terr or np.abs(fit[i,j,2])>max_tau:
                tau[i,j] = fill
            else:
                tau[i,j] = np.abs(fit[i,j,2])
    return tau
# end filter_delay_cube

'''
Filters a Space-Delay cube with a biexponential fit

$t is the time delay, and fit is the fit to a symmetric exponential

kwargs:

$fill is the tau value to fill the points that are filtered out, default (None) fills with np.nan

$min_Amp is the minimum amplitude of the fit function (sum of A,B,C), points less that this are filtered out

$max_slow is the maximum slow timeconstant, points equal to or above it are filtered out

$fracUC is the maximum fractional uncertainty in either tau_fast or tau_slow, if either have a greater
fractional uncertainty they will be filtered out

returns filtered tau values

'''
def filter_biexp_delay_cube(t, fit, fill=None, min_Amp=1.0, max_slow=100, fracUC=2.0):
    if fill is None:
        fill = np.nan
    rows, cols, N = np.shape(fit)
    tauSlow = np.zeros((rows, cols))
    tauFast = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.abs(fit[i,j,7]/fit[i,j,2])>fracUC or np.abs(fit[i,j,9]/fit[i,j,4])>fracUC or np.abs(fit[i,j,2]) > max_slow:
                tauSlow[i,j] = fill
                tauFast[i,j] = fill
            else:
                tauSlow[i,j] = np.abs(fit[i,j,2])
                tauFast[i,j] = np.abs(fit[i,j,4])
    return tauSlow, tauFast
# end filter_delay_cube

'''
Takes a fit matrix (from a power data cube) and filters the gamma values (gamma = fit[:,:,2]) based on the
sign of the slope (slovbe B = fit[:,:,1]). Returns two matirices, with values of positive slope
and one with values of negative slope.

$mingamma is the minimum absolute value of gamma needed for a point to be considered, if if
doesn't meet this threshold it will not be in either returned matrix.

If $nanfill is true will fill the empty points in both arrays with NaNs for transparent plotting
'''
def filter_slopes_power_cube(fit, mingamma=0.0, nanfill=False):
    g = fit[:,:,1]
    B = fit[:,:,0]
    rows, cols, dim = fit.shape
    if dim != 4:
        s = "Error filter_slopes_power_cube: input fit cube must have dimension of 6, based on the"
        s += "standard power law fit."
    if nanfill:
        g_above = np.empty((rows,cols))
        g_below = np.empty((rows,cols))
        g_above[:,:] = np.nan
        g_below[:,:] = np.nan
    else:
        g_above = np.zeros((rows,cols))
        g_below = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if np.abs(g[i,j]) > mingamma:
                if B[i,j] > 0.0:
                    g_above[i,j] = g[i,j]
                else:
                    g_below[i,j] = g[i,j]
    return g_above, g_below
# end filter_slopes_power_cube

'''
Derivative of Cube_Point_Display for photocurrent power cubes
'''
class Power_PCI_Cube_Point_Display(Cube_Point_Display):
	def __init__(self, rn, d, power, gamma, fit_pci, vrng=None):
		pl = lambda x, A, g, I0 : power_law(x-np.min(x), A, g, I0)
		Cube_Point_Display.__init__(self, rn, d, power, gamma, fit_pci, pl,
			xlabel='Microns',
			ylabel='Microns',
			zlabel='Power (mW)',
			vlabel='|I| (nA)',
			title=rn+' Photocurrent '+r'$\gamma$',
			figtitle='pci'
			)
		if vrng is not None:
			self.img.set_clim(vmin=vrng[0], vmax=vrng[1])
		self.axf.set_xlabel('')
		self.axf.set_ylabel('')
		self.fig.canvas.draw()
	# end init

	def get_title(self):
		return r"$\gamma = $ "+ str(round(self.fit[self.y,self.x,1],3)) + '$\pm$' + str(round(self.fit[self.y,self.x,4],2))
# end Power_PCI_Cube_Point_Display

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
