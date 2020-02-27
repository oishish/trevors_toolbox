'''
calibration.py

A module for calibration functions

Last updated December 2017

by Trevor Arp
'''
import numpy as np
from toolbox.utils import get_locals, date_from_rn
from datetime import date
import os

local_values = get_locals()
InGaAs_calibration_file = local_values['InGaAs_calibration_file']
Power_calibration_dir = local_values['Power_calibration_dir']


def current_amplifier(data, log, scale=1.0e9):
    '''
    Calibration from current amplifiers, a pre-amp and an SRS lock-in amplifier (if present).

    Args:
        data (np.ndarry) : The data image (or cube) to calibrate
        log (dict) : The log file dictionary for the run.
        unit (:obj:'float', optional) : The scale of Amperes to calibrate to. Default is 1e9, i.e. nanoamps.

    '''
    gain = log['Pre-Amp Gain']
    if 'Lock-In Gain' in log:
        gain = gain*log['Lock-In Gain']/1000.0
    return data*gain*scale
#

def power_from_meter(data, log):
    '''
    Calibrates the power based on the most recent (before the run number date) reading of a reference power meter.

    Args:
        data (np.ndarry) : The data image (or cube) to calibrate
        log (dict) : The log file dictionary for the run.
    '''
    return calibrate_power_all(log['Run Number'], data, log['Wavelength'], display=False)
# end power_from_meter

'''
Returns the responsivity factor the responsivity for a single given parameter, data should be
divided by the responsivity

For example, for the InGaAs detector responsivity varies with wavelength, so this divides by
the responsivity for that wavelength

Linear interpolation between the four nearest data points, if point is not in calibration file

Calibration file should contain two columns, the first is the parameter (sorted) the second is the responsivity
'''
def calib_responsivity(param, calibration=InGaAs_calibration_file):
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
    return float(resp)
#

'''
Calibrates data by dividing by the responsivity for a single given parameter, wrapper for
calib_resposivity

For example, for the InGaAs detector responsivity varies with wavelength, so this divides by
the responsivity for that wavelength

Linear interpolation between the four nearest data points, if point is not in calibration file

Calibration file should contain two columns, the first is the parameter (sorted) the second is the responsivity
'''
def calib_response(data, param, calibration=InGaAs_calibration_file):
    resp = calib_responsivity(param, calibration=calibration)
    return data/resp
# end calib_response

'''
Calibrates a power image or cube into a one or 2D array useful for fitting

$rn is the run number of the scan, used to find the approprate calibration file

$power is the power image or power cube of raw measured power

$wavelength of the laser

$geo_accept the geometric acceptance, percentage of how much of the measured power couples into
the GRIN lens, depreciated after improvements to optics made beam coupling tighter

returns:
A 1D (or 2D for a cube) array containing the calibrated power in mW
'''
def calibrate_power(rn, power, wavelength, geo_accept=0.35, display=True):
    files = os.listdir(Power_calibration_dir)
    rundate = date_from_rn(rn)
    lastdate = date(2014,1,1)
    for f in files:
        if f.split('.')[1] == 'txt':
            fdate = date_from_rn(f)
            if rundate >= fdate and fdate >= lastdate:
                calib_file = f
                lastdate = fdate
            if rundate >= date(2016,10,7):
                geo_accept = 1.0
            #
        #
    if display:
        print("Using Power Calibration File: " + calib_file)
    d = np.loadtxt(os.path.join(Power_calibration_dir, calib_file))
    fit = np.polyfit(d[:,1], d[:,0], 1)
    fit[0] = fit[0]*geo_accept
    fit[1] = fit[1]*geo_accept

    resp = calib_responsivity(wavelength)
    resp1250 = calib_responsivity(1250.0)

    # Average Power
    if len(power.shape) > 2:
        rows, cols, N = power.shape
        dr, dc = np.shape(d)
        if dc > 2:
            bg = np.mean(d[:,2])*np.ones(rows)
        else:
            bg = 0.0
        p = np.zeros((rows,N))
        for i in range(N):
            p[:,i] = np.mean(power[:,:,i], axis=1) - bg
            p[:,i] = p[:,i]*(resp1250/resp)
            p[:,i] = fit[0]*p[:,i] + fit[1]
    else:
        rows, cols = power.shape
        p = np.mean(power, axis=1)
        p = p*(resp1250/resp)
        p = fit[0]*p + fit[1]
    return p
# end calibrate_power

'''
Calibrates a power image or cube into a one or 2D array useful for fitting NO AVERAGING

$rn is the run number of the scan, used to find the approprate calibration file

$power is the power image or power cube of raw measured power

$wavelength of the laser

$geo_accept the geometric acceptance, percentage of how much of the measured power couples into
the GRIN lens, depreciated after improvements to optics made beam coupling tighter

returns:
The imput array calibrated into mW
'''
def calibrate_power_all(rn, power, wavelength, geo_accept=0.35, display=True):
    files = os.listdir(Power_calibration_dir)
    rundate = date_from_rn(rn)
    lastdate = date(2014,1,1)
    for f in files:
        if f.split('.')[1] == 'txt':
            fdate = date_from_rn(f)
            if rundate >= fdate and fdate >= lastdate:
                calib_file = f
                lastdate = fdate
            if rundate >= date(2016,10,7):
                geo_accept = 1.0
            #
        #
    if display:
        print("Using Power Calibration File: " + calib_file)
    d = np.loadtxt(os.path.join(Power_calibration_dir, calib_file))
    fit = np.polyfit(d[:,1], d[:,0], 1)
    fit[0] = fit[0]*geo_accept
    fit[1] = fit[1]*geo_accept
    resp = calib_responsivity(wavelength)
    resp1250 = calib_responsivity(1250.0)
    p = power*(resp1250/resp)
    p = fit[0]*p + fit[1]
    return p
# end calibrate_power_all
