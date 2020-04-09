'''
fitting.py

A module for fitting data

Last updated March 2020

by Trevor Arp

'''
import numpy as np

from gaborlab.fitting import generic_fit
from gaborlab.processing import lowpass

'''
A symmetric bi-exponential function with a fast and slow component

y = A + B*Exp(-|x/tau_slow|) + C*Exp(-|x/tau_fast|)
'''
def biexponential(x, A, B, tauS, C, tauF, t0):
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF))
# end biexponential

'''
A symmetric bi-exponential function with a fast and slow component. With a penalization parameter
to guarrentee that the fast and slow components are the correct order. Only needs to be used
for the fit
'''
def biexponential_pen(x, A, B, tauS, C, tauF, t0):
    if 1.5*tauF > tauS:
        penalization = 10000.0*tauF
    else:
        penalization = 0.0
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF)) + penalization
# end biexponential_pen

'''
Logarithmic Power Dependence Function, analytic solution to exciton rate equations

y = (1/A)*log(1+Ax) + I0
'''
def log_analytic(x, B, A, I0):
    return B*np.log(1+A*x) + I0
# end log_analytic

def biexponential_fit(x, y, p0=-1, warn=True):
    '''
    Fits data to a biexponential function.

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent varaible
        p0 (optional): The initial parameters for the fitting function. If -1 (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p : The fitting parameters that optimize the fit, or the initial paramters if the fit failed.
        perr : Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.biexponential_fit: X and Y data must have the same length")
        return
    if p0 == -1:
        ts_start = np.abs(x[0] - x[l-1])/2.0
        tf_start = ts_start/50.0
        p0 = (np.min(y), np.max(y)/10.0, ts_start, np.max(y), tf_start, 0.0)
    return generic_fit(x, y, p0, biexponential_pen, warn=warn)
# end biexponential_fit

def log_analytic_fit(x, y, p0=-1, warn=True):
    '''
    Fits data to to the logarithm-analytic function.

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent varaible
        p0 (optional): The initial parameters for the fitting function. If -1 (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p : The fitting parameters that optimize the fit, or the initial paramters if the fit failed.
        perr : Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.log_analytic_fit: X and Y data must have the same length")
        return
    if p0 == -1:
        p0=(np.mean(y), 0.1, 0.0)
    return generic_fit(x, y, p0, log_analytic, warn=warn)
# end log_analytic_fit

def mean_residual(data, fit):
    '''
    A generic function of calculating mean absolute value of the residual

    Args:
        data : the data to calculate
        fit : the corresponding values of the fitted function
    '''
    return np.mean(np.abs(data-fit))
# end mean_residual

'''
Takes a data cube and lowpases the columns of each scan using fitting.lowpass
'''
def lp_cube_cols(datacube, cutoff=0.05, samprate=1.0):
    rows, cols, N = datacube.shape
    original = np.copy(datacube)
    for j in range(N):
        for i in range(cols):
            datacube[:,i,j] = lowpass(original[:,i,j], cutoff=cutoff, samprate=samprate)
    return datacube
# end lp_cube_cols

'''
Takes a data cube and lowpases the rows, then the columns of each scan using fitting.lowpass
'''
def lp_cube_rows_cols(datacube, cutoff=0.05, samprate=1.0):
    rows, cols, N = datacube.shape
    original = np.copy(datacube)
    for j in range(N):
        for i in range(rows):
            datacube[i,:,j] = lowpass(original[i,:,j], cutoff=cutoff, samprate=samprate)
        for i in range(cols):
            datacube[:,i,j] = lowpass(datacube[:,i,j], cutoff=cutoff, samprate=samprate)
    return datacube
# end lp_cube_cols

'''
Takes a 2D scan and lowpases the columns of each scan using fitting.lowpass
'''
def lp_scan_cols(data, cutoff=0.05, samprate=1.0):
    rows, cols = data.shape
    original = np.copy(data)
    for i in range(cols):
        data[:,i] = lowpass(original[:,i], cutoff=cutoff, samprate=samprate)
    return data
# end lp_cube_cols

'''
Takes a data cube and subtracts out the background from each individual scan,
determines the background from the values of the last $nx columns

$ix is the number of columns at the end of each row to use as background
'''
def subtract_bg_cube(datacube, nx=20):
    rows, cols, N = datacube.shape
    for j in range(N):
        n = np.mean(datacube[:,cols-nx:cols,j], axis=1)
        for i in range(rows):
            datacube[i,:,j] = datacube[i,:,j] - n[i]
    return datacube
# end subtract_bg_cube
