'''
A place for things that interface with the older versions of the DAQ software,
and older versions of my code more generally.
'''
import numpy as np
from os.path import exists, join

from gaborlab.mpdpm import Run
from gaborlab.utilities import find_run, find_savefile
import standards as st


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
