'''
mpdpm.py

A module for processing MPDPM as objects

Last updated: February 2020

by Trevor Arp
'''
from os.path import exists, join
import pickle

import numpy as np
from scipy.ndimage.interpolation import shift as ndshift

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
        shape : the shape of the data images, similar to numpy.shape
    '''
    def __init__(self, run_num, autosave=True, overwrite=False, calibrate=None, stabalize=True, preprocess=None, customdir=None):
        '''
        Args:
            run_num (str) : The run number in standard hyperDAQ form, i.e. "SYSTEM_YEAR_MONTH_DAY_NUMBER"
            autosave (:obj:'bool', optional) : Whether or not to save a processed file after processing is complete. Defaults to True.
            overwrite (:obj:'bool', optional) : If True will load noramlly and overwrite a previous processed savefile during autosave. Defaults to False.
            calibrate (:obj:'dict', optional) : If not None will use value as the calibration specification. By default (None) uses calib_spec in standards.py.
            stabalize (:obj:'bool', optional) : If True will stabalize a spatial cube according to the standard_cube_stabalization in standards.py. Defaults to True.
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

        '''
        If it has a savefile in the processed directory, load the data and log from it
        '''
        if not overwrite:
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
            if output in self.log['Fast Axis Variable']:
                self.log[output] = (self.log['Fast Axis Start'], self.log['Fast Axis End'])
            elif output in self.log['Slow Axis Variable']:
                self.log[output] = (self.log['Slow Axis Start'], self.log['Slow Axis End'])
            elif 'Cube Axis' in self.log and output in self.log['Cube Axis']:
                self.log[output]  = (self.log['Cube Axis Start'], self.log['Cube Axis End'])
            else:
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
            if axlbls[i] == "Cube": # For some reason the cube axis labeling is a bit different
                varlbl = ' Axis'
            else:
                varlbl = ' Axis Variable'
            if self.log[axlbls[i] + varlbl] in st.measured_axes: # If it is a measured axis
                img = st.measured_axes[self.log[axlbls[i] + varlbl]]
                if axlbls[i] == "Cube":
                    ind = [0,1,2]
                else:
                    ind = [0,1]
                ind.remove(i)
                self.axes.append(np.mean(self.data[img], axis=tuple(ind))) # Average along the other axes
            else: # If it is a sampled axis
                if self.log[axlbls[i] +' Axis Sampling'] in st.sampling:
                    samplefunc = st.sampling[self.log[axlbls[i] +' Axis Sampling']]
                else:
                    raise KeyError("Sampling function " + self.log[axlbls[i] +' Axis Sampling'] + " not in standards.py")
                self.axes.append(samplefunc(self.log[axlbls[i] +' Axis Start'], self.log[axlbls[i] +' Axis End'], self.shape[i]))
        #

        # Stabalize the images if needed
        if stabalize:
            self._stabalize()
        #

        # Save the processed file
        if autosave or overwrite:
            self.save(directory=customdir)
        #
    # end __init__

    def get(self):
        '''
        Returns the data images (i.e. *data) in alphabetical order of keys
        '''
        output = []
        for k in sorted(self.data):
            output.append(self.data[k])
        return output
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

    def save(self, directory=None):
        '''
        Saves the images to the processed directory.

        Args:
            directory (str, optional): A custom save directory, if None it will search the processed directory defined in the locals file
        '''
        savefile = find_savefile(self.run_number, directory=directory)
        with open(join(savefile, self.run_number+"_log.pkl"),'wb') as fl:
            pickle.dump(self.log, fl)
        np.savez(join(savefile, self.run_number+"_run.npz"), axes=self.axes, units=self.units, shape=self.shape, **self.data)
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
                    self.data[k][:,:,i] = func(self.data[k][:,:,i])
            else:
                self.data[k] = func(self.data[k])
    # end process

    def _load(self, savefile):
        '''
        Loads the images from the processed directory
        '''
        files = np.load(join(savefile, self.run_number+"_run.npz"))
        try:
            with open(join(savefile, self.run_number+"_log.pkl"),'rb') as fl:
                self.log = pickle.load(fl)
            self.axes = files['axes']
            self.units = files['units']
            self.shape = files['shape']
            self.data = {}
            types = self.log['Data Files']
            types = types.split(',')
            for s in types:
                self.data[s] = files[s]
            return 0
        except Exception:
            print("DEBUG")
            return -1
    # end save

    def _stabalize(self,):
        '''
        Stabalize the images if the Run is a spatial data cube
        '''
        # [image_extenstion, stabalization_function, last]
        if self.units[0] in st.spatial_units and self.units[1] in st.spatial_units and len(self.shape) == 3: # If it is a spatial cube
            key, func, last = st.standard_cube_stabalization
            rows, cols, N = self.shape
            if last:
                ref = self.data[key][:,:,N-1]
            else:
                ref = self.data[key][:,:,0]
            for i in range(0, N-1):
                sft = func(self.data[key][:,:,i], ref)
                for k in self.data.keys():
                    self.data[k][:,:,i] = ndshift(self.data[k][:,:,i], sft, cval=np.mean(self.data[k][:,:,i]))
        #
    # end stabalize
# end Run


class DataSet(Run):
    '''
    Should have all of the attributes and core functions of a run but builds up from multiple runs.

    Have a lookup file of datasets, loade them in, haev the same attributes as Run (stiched together from many)
    and then have an items attribute that contains the indiviual runs

    If possible make recursive so that you can have a set of sets
    '''
    def __init__(self, setname):
        pass
    # end __init__

    def save(self):
        '''
        Saves in it's own tree, maybe with a checksum?
        '''
        pass
    # end save
# end DataSet
