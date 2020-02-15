'''
mpdpm.py

A module for processing MPDPM as objects

Last updated: February 2020

by Trevor Arp
'''
from os.path import exists, join
import numpy as np
from toolbox.utils import find_run

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
    def __init__(self, run_num, autosave=True, overwite=False, calibrate=True, stabalize=True, preprocess=None, customdir=None):
        '''
        Args:
            run_num (str) : The run number in standard hyperDAQ form, i.e. "SYSTEM_YEAR_MONTH_DAY_NUMBER"
            autosave (:obj:'bool', optional) : Whether or not to save a processed file after processing is complete. Defaults to True.
            overwrite (:obj:'bool', optional) : If True will overwrite a prvious processed savefile during autosave. Defaults to False.
            calibrate (:obj:'bool', optional) : If True will calibrate according to the calibrate() function. Defaults to True.
            stabalize (:obj:'bool', optional) : If True will stabalaize the images (if possible) according to the stabalize() function. Defaults to True.
            preprocess (:obj:'dict', optional) : If not None will call the process function with the value as the argument
            customdir (str, optional) : A custom save directory, if None it will search the directory defined in the locals file
        '''
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
            if self.log[axlbls[i] +' Axis Sampling'] == 'linspace':
                samplefunc = np.linspace
            else:
                raise ValueError("Sampling function not yet implemented")
                '''
                !!!!!!!!!!!!!!!!!!!!
                # Need a way to pick out the sampling function for non-standard sampling.
                # Maybe some kind of namespace file akin to the locals? This would also help with the
                # other things that need defaults
                '''
            self.axes.append(samplefunc(self.log[axlbls[i] +' Axis Start'], self.log[axlbls[i] +' Axis End'], self.shape[i]))
        #

        # Calibrate the values as specified in the default.

        # Calibrate various axes

        # Stabalize if needed

    # end __init__

    def get(self):
        '''
        Returns the data images i.e. *data in alphabetical order of keys
        '''
        pass
    # end get

    def convert(self):
        '''
        Takes an axis and converts the units
        '''
        pass
    # end convert

    def shape(self):
        '''
        Returns the shape of the data images, simiplar to numpy.shape
        '''
        return self.shape
    # end shape

    def calibrate(self):
        '''
        Calibrates all the defined axes.
        ##### Needs to have some standard method of calibrating, and some options
        '''
        pass
    # end calibrate

    def stabalize(self):
        '''
        Stabalized images if possible
        ###### Needs to have some default method of calibrating
        '''
        pass
    # end stabalize

    def save(self):
        '''
        Saves the images to the processed directory
        '''
        pass
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
        pass
    # end save
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
