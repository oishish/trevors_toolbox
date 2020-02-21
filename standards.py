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
import toolbox.fitting as fitting


# Sampling is a dictionary where the keys are the log file names for a sampling function and the
# values are the function. To be used to get the values when loading an axis.
sampling = {'linspace':np.linspace}

# A dictionary of the axes where thel values are measured, not sampled. If an axis has a given
# name (key) it's values will be dervied from the immage corresponding to the extension (value).
measured_axes = {'Power (%)':'pow'}

# Constant card outputs that are always recorded as variables. Will generate a log entry with this
# key, no matter what these varaibles were set to in the scan
card_ouput_entries = ['Source/Drain', 'Backgate']

# The standard calibration functios to use on the data images
# Calibration functions take arguments calib_function(dataimage, log_file_dict)
# If None that axis will not be calibrated
standard_image_calibration = {'rfi':None, 'pci':calibration.current_amplifier, 'pow':calibration.power_from_meter}

# Standard Data Cube stabalization parameters
# [image_extenstion, function, last]
# where function is a function that takes func(image, ref_image) and computes the shift that maps image onto ref_image
# where last is a boolean, if True will stabalize to the last scan in the cube, otherwise will stabalize to the first scan
standard_cube_stabalization = ['pci', fitting.norm_compute_shift, True]

# The units that are considered spatial for the purposes of stabalization
spatial_units = ['Micron']
