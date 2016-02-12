'''
toolbox.py

This module provides a common set of functions and namespace for analysis scripting. It is designed
to be used with a wild import, i.e. from dimage import *, to provide a namespace.

It imports various sub-modules for acess and a few common functions from them are imported directly
intro the namespace. For example the display module contains a lot of plotting functions, most must
be referenced as display.<module> but get_viridis() and format_plot_axes() are imported directly
due to their common usage.

It also defines the core functions for acessing data sets.

Last updated February 2016
by Trevor Arp
'''

'''
Generally Usefull Modules
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''
Utilities module contains commonly used general purpose functions.
'''
import utils

'''
Functions for manipulating scans, a few common functions are directly imported
'''
import scans
slowscan_2_line = scans.slowscan_2_line
range_from_log = scans.range_from_log


'''
Contains functions relating to displaying data runs, a few common functions are directly imported
'''
import display
set_img_ticks = display.set_img_ticks
format_plot_axes = display.format_plot_axes
get_viridis = display.get_viridis

'''
Calibration functions
'''
import calibration

'''
Fitting functions
'''
import fitting

'''
Simulation functions
'''
import sim

'''
Functions for processing runs generated by various instruction sets
'''
import process
load_run = process.load_run

'''
Functions for processing runs generated by various instruction sets
'''
import postprocess

'''
A object for plotting a data image, takes a run number and plots

$run_num and $dir are the data set to open, obeying the same conventions as load_run
'''
class dataimg():
    def __init__(self, run_num, directory=''):
        l, d = load_run(run_num, directory)
        self.log = l
        for k,v in d.items():
            setattr(self,k,v)
        self.run_num = run_num
        self.shape = np.shape(self.pci)
    # end init
# end dataimg
