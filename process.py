'''
process.py

A module for importing and processing data runs generated by various instruction sets

Last updated February 2016

by Trevor Arp
'''
import numpy as np
from os.path import exists, isfile
from utils import indexof
from timeit import default_timer as timer
import datetime

from fitting import power_law_fit
from fitting import symm_exponential_fit as get_delay_single_fit
from fitting import double_exponential_fit as get_delay_double_fit
from fitting import lowpass

from calibration import calib_response

from defaults import *

'''
Takes a run number and loads the run including the reflection image, the photocurrent image
(which may be an image or a data cube) and the contents of the log file in a dictionary.


 returns log, data
 where data is a dictionary of the data

$directory is the directory, either absolute path to the directory or the subdirectory within datadir
'''
def load_run(run_num, directory=''):
    # Find the file
    if exists(run_num + '_log.log' ):
        path = run_num
    elif exists(directory + '\\' + run_num + '_log.log'):
        path = directory + '\\' + run_num
    elif exists(datadir + run_num + '_log.log'):
        path = datadir + run_num
    elif exists(datadir + directory + '\\' + run_num + '_log.log'):
        path = datadir  + directory + '\\' + run_num
    else:
        print directory + '\\' + run_num + '_log.log'
        print exists(directory + '\\' + run_num + '_log.log' )
        print 'Error dimage.load_run : Could not open files'
        raise IOError
        #

    f = open(path+'_log.log', 'r').readlines()

    # For backwards compatability
    if f[0] == "##### Scan Paramters #####\n":
        log, data = old_load_run(run_num, directory=directory)
        return log, data

    # Load the log file
    log = {}
    for line in f:
        s = line.split(':')
        if len(s) == 2:
            k = s[0]
            v = s[1]
            try:
                log[k] = float(v)
            except ValueError:
                log[k] = str(v).strip()
        elif len(s) > 2:
            k = s[0]
            v = s[1:len(s)]
            log[k] = str(v).strip()
    log['Fast Axis'] = (log['Fast Axis Start'], log['Fast Axis End'])
    log['Slow Axis'] = (log['Slow Axis Start'], log['Slow Axis End'])
    log['Source/Drain'] = (log['Source/Drain Start'], log['Source/Drain End'])
    log['Backgate'] = (log['Backgate Start'], log['Backgate End'])

    #load the data
    data = {}
    types = log['Data Files']
    types = types.split(',')
    for s in types:
        if exists(path + '_' + s +'.dat'):
            data[s] = np.loadtxt(path + '_' + s +'.dat')
        elif exists(path + '_' + s +'.npy'):
            data[s] = np.load(path + '_' + s +'.npy')
        else:
            "Error in load_run: Cannot find data file for filetype: " + str(s)
            raise IOError
    return log, data
# end load_run

'''
load_runs for files taken before software version 3.2
'''
def old_load_run(run_num, directory=''):

    # Find the file
    if exists(run_num + '_log.log' ):
        path = run_num
    elif exists(directory + '\\' + run_num + '_log.log'):
        path = directory + '\\' + run_num
    elif exists(datadir + run_num + '_log.log'):
        path = datadir + run_num
    elif exists(datadir + directory + '\\' + run_num + '_log.log'):
        path = datadir  + directory + '\\' + run_num
    else:
        print directory + '\\' + run_num + '_log.log'
        print exists(directory + '\\' + run_num + '_log.log' )
        print 'Error dimage.load_run : Could not open files'
        raise IOError
        #

    # Load the data
    data = {}
    if exists(path + '_rfi.dat'):
        data['rfi'] = np.loadtxt(path + '_rfi.dat')
    elif exists(path + '_rfi.npy'):
        data['rfi'] = np.load(path + '_rfi.npy')

    if exists(path + '_pow.dat'):
        data['pow'] = np.loadtxt(path + '_pow.dat')
    elif exists(path + '_pow.npy'):
        data['pow'] = np.load(path + '_pow.npy')

    if exists(path + '_wav.dat'):
        data['wav'] = np.loadtxt(path + '_wav.dat')
    elif exists(path + '_wav.npy'):
        data['wav'] = np.load(path + '_wav.npy')

    if exists(path + '_pci.dat'):
        data['pci'] = np.loadtxt(path + '_pci.dat')
    elif exists(path + '_pci.npy'):
        data['pci'] = np.load(path + '_pci.npy')
    elif exists(path + '_dc.npy'):
        data['pci'] = np.load(path + '_dc.npy')


    # Load the log file
    f = open(path+'_log.log', 'r').readlines()
    log = {}
    log['Date'] = f[1].strip()
    log['Run Number'] = run_num
    for line in f:
        if len(line.split(':')) == 2:
            if 'from' in line:
                l = line.split(':')
                name = str(l[0])
                l = l[1].split(' ')
                ix = indexof(l,'from')
                if not name in log:
                    if name == 'Ranging Parameter':
                        s = ''
                        for i in range(ix):
                            s = s + ' ' + str(l[i])
                            s = s.strip()
                        log['Ranging Parameter'] = s
                        name = s
                    log[name] = (float(l[ix+1]), float(l[ix+3]))
            else:
                l = line.split(':')
                if l[0]=='Comment' or l[0]=='Varying Parameter':
                    log[str(l[0])] = str(l[1])
                elif l[0] == 'Range Values':
                    from ast import literal_eval as ast_literal_eval
                    l[1] = l[1].strip()
                    log[str(l[0])] = ast_literal_eval(l[1])
                else:
                    log[str(l[0])] = float(l[1])
        elif 'Started' in line:
            log['Scan Type'] = line.split(' ')[0]

    return log, data
# end old_load_run

'''
For a map of reflection computes and returns a map of Delta R over R

$r is the reflection image,

$backgnd is the area that is considered "Background", data coordinates (x0, x1, y0, y1)
if None (default) it is an (N/10, M/10) rectangle in the lower left corner of a N by M reflection Scans
'''
def compute_drR(r, backgnd=None):
    if backgnd == None:
        rows, cols = np.shape(r)
        backgnd = (0, rows/10, 9*cols/10, cols)
    R = np.mean(r[backgnd[2]:backgnd[3],backgnd[0]:backgnd[1]])
    return (r-R)/R
# end compute drR

'''
#############################################
Instruction Set Processing Scripts
#############################################

Generic Processing for the instruction sets, name of function is same as instruction set

All functions take a run (dataimg object) as the main input and have a savefile option to save then
output to some file for fast access.
'''

'''
Processes a Finite Scan, basic calibration into nA from the gains. Also works for Continuous Scans

Parameters:
$run is the dataimg object for the input run

Returns:
calibrated data

$rfi (raw), $pci (nA)
'''
def Finite_Scan(run):
    log = run.log
    gain = log['Pre-Amp Gain']*(log['Lock-In Gain']/1000.0)
    pci = run.pci*(gain*1.0e9)
    return run.rfi, pci
# end Finite Scan

'''
Processes a Delay Slow Scan, Line scan along the fast axis, delay along the slow axis.

Computes signal from the maximum of the lowpassed photocurrent for each linescan (row)

Parameters:
$run is the dataimg object for the input run

$start and $end are the row indicies to fit between, i.e. data[start:row] will be fit. Default
is 0 and # of rows respectively.

Returns:

$t the two-pulse time delay

$backgnd the background photocurrent, average of the last 10 columns of the scan

$signal is the photocurrent signal, computed from the lowpassed maxima of each linescan

$diff the difference between the signal and background

$params is the fitted parameters, corresponsing to fitting.symm_exp

$perr is the error in the fitting parameters $params

returnt, backgnd, signal, diff, params, perr

'''
def Delay_Slow_Scan(run, start=0, end=None):
    log = run.log
    gain = log['Pre-Amp Gain']*(log['Lock-In Gain']/1000.0)
    d = run.pci*(gain*1.0e9)
    power = run.pow
    ref = run.rfi
    rows, cols = d.shape

    if end is None:
        end = rows

    t = range_from_log('Delay (ps)', log, rows)

    d = d[start:rows,:]
    power = power[start:rows,:]
    ref = ref[start:rows,:]
    t = t[start:rows]
    rows, cols = d.shape

    backgnd = np.zeros(rows)
    signal = np.zeros(rows)
    signalix = range(rows)
    diff = np.zeros(rows)
    dpow = np.zeros(rows)
    dref = np.zeros(rows)
    for i in range(rows):
        l = lowpass(d[i,:])
        signalix[i] = np.argmax(l)
        backgnd[i] = np.mean(l[cols-10:cols])
        signal[i] = l[signalix[i]]
        dpow[i] = power[i,signalix[i]]
        dref[i] = ref[i,signalix[i]]
        diff[i] = signal[i] - backgnd[i]

    params, perr = get_delay_single_fit(t, diff)
    return t, backgnd, signal, diff, params, perr
# end Delay Slow Scan

'''
Processes a Rotation Scan with a linescan along the fast axis and Rotating an attenuator along
the Slow Axis

returns
'''
def Full_Rotation_Scan(run, savefile=None):
    pass
# end Fast Piezo Slow Delay

'''
Processes a scan varying piezo vcoltage along the fast axis and Delay along the slow axis

returns
'''
def Fast_Piezo_Slow_Delay(run, savefile=None):
    pass
# end Fast Piezo Slow Delay

'''
Retrives, averages the power and fits a data cube containing spatial scans with varying power.

For the fit, takes each point and the given power and fits a power law

where $fit is a 3D array where for each point in the input map, it has the fit parameters and the
fit errors for a power law fit as fit[i,j,:] = [A, B, g, A_error, B_error, g_error] for the fit
function:
y = A + B*x^g

Parameters:
$run is the dataimg object for the input run

$savefile is the place to save the processed data to, or load from if it already exists. If None
(default) does not save.

$geometic_calib is the rought geometric conversion to milliwatts, based on alignment, measure separatly and
don't trust absolutely, linear: power = geometric_calib[0]*raw_power + geometric_calib[1]

$backgnd is the background area for Delta R over R, same spec as in the compute_drR function

$default and $default_err are the values to default to if the fitting routine fails.

$display When processing timing info is printed to terminal

Returns:
returns the averaged power, dR/R, photocurrent and the fits as

$power, $drR, $pci, $fit_drR, $fit_pci
'''
def Space_Power_Cube(run,
    savefile=None,
    geometric_calib=(3.42, 0.284),
    backgnd=None,
    default=(-1,-1,-1),
    err_default=(-1,-1,-1.0),
    display=True ):
    log = run.log
    rn = log['Run Number']
    gain = log['Pre-Amp Gain']*(log['Lock-In Gain']/1000.0)
    d = run.pci*(gain*1.0e9)
    p = run.pow
    wavelength = round(run.log['Wavelength'])
    rows, cols, N = d.shape
    power = np.zeros((rows,N))

    # Compute delta R over R
    r = run.rfi
    drR = np.zeros((rows,cols, N))
    for i in range(N):
        drR[:,:,i] = compute_drR(r[:,:,i], backgnd)

    # Average Power
    for i in range(N):
        for j in range(rows):
            power[j, i] = np.mean(p[j,:,i])
        # Calibrate
        power[:,i] = calib_response(power[:,i], wavelength)
        power[:,i] = geometric_calib[0]*power[:,i] + geometric_calib[1]

    # If it hasn't already been saved to the savefile
    condition = savefile is not None and exists(savefile + rn + "_fitpci.npy")
    condition = condition and exists(savefile + rn + "_fitrfi.npy")
    if condition:
        fit_drR = np.load(savefile + rn + "_fitrfi.npy")
        fit_pci = np.load(savefile + rn + "_fitpci.npy")
    else:
        if display:
            s = str(rows) + 'x' + str(cols) + 'x' + str(N)
            print "Starting Processing on " + s + " datacube"
        t0 = timer()
        fit_drR = np.zeros((rows, cols, 2*N))
        fit_pci = np.zeros((rows, cols, 2*N))
        for i in range(rows):
            for j in range(cols):
                params, err = power_law_fit(power[i,:], np.abs(d[i,j,:]), p_default=default, perr_default=err_default)
                fit_pci[i,j,0:3] = params
                fit_pci[i,j,3:6] = err
                params, err = power_law_fit(power[i,:], np.abs(drR[i,j,:]), p_default=default, perr_default=err_default)
                fit_drR[i,j,0:3] = params
                fit_drR[i,j,3:6] = err
        tf = timer()
        if display:
            print " "
            dt = tf-t0
            print "Processing Completed in: " + str(datetime.timedelta(seconds=dt))
        if savefile is not None:
            np.save(savefile + rn + "_fitpci.npy", fit_pci)
            np.save(savefile + rn + "_fitrfi.npy", fit_drR)
    return power, drR, d, fit_drR, fit_pci
# end fit_power_cube
