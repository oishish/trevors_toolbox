'''
calibration.py

A module for calibration functions

Last updated April 2016

by Trevor Arp
'''
import numpy as np
from utils import get_locals, date_from_rn
import os

local_values = get_locals()
InGaAs_calibration_file = local_values['InGaAs_calibration_file']
Power_calibration_dir = local_values['Power_calibration_dir']

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

'''
Calibrates a power image or cube into a one or 2D array useful for fitting

$rn is the run number of the scan, used to find the approprate calibration file

$power is the power image or power cube of raw measured power

$wavelength of the laser

$geo_accept the geometric acceptance, percentage of how much of the measured power couples into
the GRIN lens

returns:
A 1D (or 2D for a cube) array containing teh calibrated power in mW

'''
def calibrate_power(rn, power, wavelength, geo_accept=0.35):
    files = os.listdir(Power_calibration_dir)
    rundate = date_from_rn(rn)
    for f in files:
        if f.split('.')[1] == 'txt':
            fdate = date_from_rn(f)
            if rundate > fdate:
                calib_file = f
            #
        #
    print "Using Power Calibration File: " + calib_file
    d = np.loadtxt(os.path.join(Power_calibration_dir, calib_file))
    fit = np.polyfit(d[:,1], d[:,0], 1)
    fit[0] = fit[0]*geo_accept
    fit[1] = fit[1]*geo_accept


    # Average Power

    if len(power.shape) > 2:
        rows, cols, N = power.shape
        bg = np.mean(d[:,2])*np.ones(rows)
        p = np.zeros((rows,N))
        for i in range(N):
            p[:,i] = np.mean(power[:,:,i], axis=1) - bg
            p[:,i] = calib_response(p[:,i], wavelength)
            p[:,i] = fit[0]*p[:,i] + fit[1]
    else:
        rows, cols = power.shape
        p = np.mean(power, axis=1)
        p[:,i] = calib_response(p[:,i], wavelength)
        p[:,i] = fit[0]*p[:,i] + fit[1]
    return p
# end calibrate_power
