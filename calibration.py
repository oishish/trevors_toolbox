'''
calibration.py

A module for calibration functions

Last updated February 2016

by Trevor Arp
'''
import numpy as np
from utils import get_locals

local_values = get_locals()
InGaAs_calibration_file = local_values['InGaAs_calibration_file']

'''
Calibrates data by dividing by the responsivity for a single given parameter

For example, for the InGaAs detector responsivity varies with wavelength, so this divides by
the responsivity for that wavelength

Linear interpolation between the four nearest data points, if point is not in calibration file

Calibration file should contain two columns, the first is the parameter (sorted) the second is the responsivity
'''
def calib_response(data, param, calibration=InGaAs_calibration_file):
    c = np.loadtxt(calibration)
    rows, cols = np.shape(c)
    param = float(param)
    if param not in c[:,0]:
        ix = np.searchsorted(c[:,0], param)
        if ix < 2:
            resp = np.interp(param, c[0:4,0], c[0:4,1])
        elif ix > rows-2:
            resp = np.interp(param, c[rows-4:rows,0], c[rows-4:rows,1])
        else:
            resp = np.interp(param, c[ix-2:ix+2,0], c[ix-2:ix+2,1])
    else:
        i = int(np.argwhere(c[:,0]==param))
        resp = c[i,1]
    return data/float(resp)
#
