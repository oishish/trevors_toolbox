'''
standards.py

A module defining standard functions to use in the data hand various , i.e. calibration
and processing.

This is a default file from the toolbox but the default functionality can be overwritten or
extended by making a standards.py file in the local directory. To ensure compatability copy
this file to the local directory and then modify. To ensure tracability all modifications should
be commented and well documented.
'''
import numpy as np
import toolbox.calibration as calibration


# Sampling is a dictionary where the keys are the log file names for a sampling function and the
# values are the function. To be used to get the values when loading an axis.
sampling = {'linspace':np.linspace}

# A dictionary of the axes where thel values are measured, not sampled. If an axis has a given
# name (key) it's values will be dervied from the immage corresponding to the extension (value).
measured_axes = {'Power (%)':'pwi'}


# The standard calibration functios to use on the data images
# Calibration functions take arguments calib_function(dataimage, log_file_dict)
# If None that axis will not be calibrated
standard_image_calibration = {'rfi':None, 'pci':calibration.current_amplifier, 'pwi':calibration.power_from_meter}
