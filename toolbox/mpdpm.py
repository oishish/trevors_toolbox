'''
mpdpm.py

A module for processing MPDPM as objects

Last updated: February 2020

by Trevor Arp
'''
from os.path import exists, join
import numpy as np
from toolbox.utils import find_run, find_savefile
import standards as st

class Run():
    '''
    Handels a single MPDPM run with a run number. Data is loaded and (usually) calibrated and/or
    stabalized, ready for use. There are certain key attributes and core function described below.
    (Cannot process runs before 2016/2/5 when the current log file format was introduced.)

    Attributes:
        log : A dictionary of all the values in the log file
        axes : A list of numpy arrays giving all the axes values in order, i.e. [fast, slow, cube, ...]
        units : A list of the units (as strings) of all the axes, in order
        data : A dictionary of the data images, where the keys are the image extensions and the values are arrays of the data
    '''
    def __init__(self, run_num, autosave=True, overwite=False, calibrate=None, stabalize=True, preprocess=None, customdir=None):
        '''
        Args:
            run_num (str) : The run number in standard hyperDAQ form, i.e. "SYSTEM_YEAR_MONTH_DAY_NUMBER"
            autosave (:obj:'bool', optional) : Whether or not to save a processed file after processing is complete. Defaults to True.
            overwrite (:obj:'bool', optional) : If True will load noramlly and overwrite a previous processed savefile during autosave. Defaults to False.
            calibrate (:obj:'dict', optional) : If not None will use value as the calibration specification. By default (None) uses calib_spec in standards.py.
            stabalize (:obj:'bool', optional) : If True will stabalaize the images (if possible) according to the stabalize() function. Defaults to True.
            preprocess (:obj:'dict', optional) : If not None will call the process function with the value as the argument
            customdir (str, optional) : A custom save directory, if None it will search the directory defined in the locals file
        '''
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
        #

        '''
        If it has a savefile in the processed directory, load the images from it
        '''
        if not overwrite:
            savefile = find_savefile(rn, directory=directory)
            if exists(join(savefile, rn+"_run.npz")):
                r = self._load(join(savefile, rn+"_run.npz"))
                if r == 0:
                    return
                else:
                    print("Warning: could not load savefile: " + join(savefile, rn+"_run.npz") + " loading from raw data")
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
            calib = st.standard_calibration
        else:
            calib = calibrate
        for k,v in self.data.items():
            if calib[k] is not None:
                self.data[k] = calib[k](v, self.log)
        #

        # Assemble the axes
        self.axes = []
        self.units = []
        if int(self.log['Scan Dimension']) == 3: # Cube case
            axlbls = ["Fast", "Slow", "Cube"]
        else:
            axlbls = ["Fast", "Slow"]
        #

        for i in range(len(axlbls)):
            self.units.append(self.log[axlbls[i] +' Axis Units'])
            if self.log[axlbls[i] +' Axis Variable'] in st.measured_axes: # If it is a measured axis
                img = st.measured_axes[self.log[axlbls[i] +' Axis Variable']]
                ind = [0,1,2]
                ind.remove(i)
                self.axes.append(np.mean(img, axis=tuple(ind))) # Average along the other axes
            else: # If it is a sampled axis
                if self.log[axlbls[i] +' Axis Sampling'] in st.sampling:
                    samplefunc = st.sampling[self.log[axlbls[i] +' Axis Sampling']]
                else:
                    raise KeyError("Sampling function " + sfuncval + " not in standards.py")
                self.axes.append(samplefunc(self.log[axlbls[i] +' Axis Start'], self.log[axlbls[i] +' Axis End'], self.shape[i]))
        #

        # Stabalize the images if needed
        if stabalize:
            self.stabalize()
        #

        # Save the processed file
        if autosave or overwite:
            self.save(directory=customdir)
        #
    # end __init__

    def get(self):
        '''
        Returns the data images (i.e. *data) in alphabetical order of keys
        '''
        output = []
        for k in sorted(self.data):
            output.append(data[k])
        return *output
    # end get

    def convert(self, axis, factor, newunit):
        '''
        Takes an axis and converts the units.

        Args:
            axis (int) : the index of the axis to convert in Run.axes
            factor (float) : the conversion factor, equal to (new unit)/(old unit)
            newunit (str) : the name of the new unit
        '''
        self.axes[int(axis)] = factor*self.axes[int(axis)]
        self.units[int(axis)] = newunit
    # end convert

    def shape(self):
        '''
        Returns the shape of the data images, similar to numpy.shape
        '''
        return self.shape
    # end shape

    def stabalize(self):
        '''
        Stabalized images if possible
        ###### Needs to have some default method of calibrating
        '''
        raise ValueError("Need to implement stabalization")
    # end stabalize

    def save(self, directory=None):
        '''
        Saves the images to the processed directory.

        Args:
            directory (str, optional): A custom save directory, if None it will search the processed directory defined in the locals file
        '''
        savefile = find_savefile(self.run_number, directory=directory)
        np.savez(savefile, axes=self.axes, units=self.units, shape=self.shape, **self.data)
    # end save

    def _load(self, savefile):
        '''
        Loads the images from the processed directory
        '''
        files = np.load(join(savefile, rn+"_run.npz"))
        try:
            self.axes = files['axes']
            self.units = files['units']
            self.shape = files['shape']

            self.data = {}
            types = self.log['Data Files']
            types = types.split(',')
            for s in types:
                self.data[s] = files[s]
        except KeyError:
            return -1
    # end save

    def process(self, spec):
        '''
        Perform some type of processing, will be called in initialization if a process spec is
        passed to the preprocess argument.

        Args:
            spec (:obj:'dict') : A dictionary defining how to process, keys are image extensions and
                value is a function to process that image. Function should only take a single 2D image
                as an argument. For datacubes the function will be applied to each component image.
        '''
        for k,func in spec.items():
            if len(self.shape) == 2:
                for i in range(self.shape[2]):
                    data[k][:,:,i] = func(data[k][:,:,i])
            else:
                data[k] = func(data[k])
    # end process
# end Run


class DataSet(Run):
    '''
    Should have all of the attributes and core functions of a run but builds up from multiple runs
    '''
    def __init__(self, runs):
        pass
    # end __init__

    def save(self):
        '''
        Saves in it's own tree, maybe with a checksum?
        '''
        pass
    # end save
# end DataSet
