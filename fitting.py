'''
fitting.py

A module for fitting data

Last updated February 2016

by Trevor Arp

'''
import numpy as np
import scipy as sp

from scipy.optimize import curve_fit as fit
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq

'''
Symmetric Exponential Function

y = A + B*Exp(-|x/tau|)
'''
def symm_exp(x,A,B,tau):
    return A + B*np.exp(-np.abs(x/tau))

'''
Fits data $x and $y to a symmetric exponential function defined by fitting.symm_exp

Returns the fit parameters and the errors in the fit parameters as (p, perr)

Default parameters:

$p0 is the starting fit parameters,
leave as-1 to estimate starting parameters from data

$xstart is the first data point to include in the fit,
leave as -1 to start with the first element in $x and $y

$xstop is the last data point to include in the fit,
leave as -1 to start with the last element in $x and $y
'''
def symm_exponential_fit(x, y, p0=-1, xstart=-1, xstop=-1):
    l = len(y)
    if len(x) != l :
        print "Error fitting.symm_exponential_fit: X and Y data must have the same length"
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        a = np.mean(y)
        b = np.mean(y[9*l/20:11*l/20]) - a
        t = (x[xstop-1]-x[xstart] )/4.0
        p0=(a, b, t)
    try:
        p, plconv = fit(symm_exp, x[xstart:xstop], y[xstart:xstop],p0=p0)
        perr = np.sqrt(np.diag(plconv))
    except Exception as e:
        p = p0
        perr = (0,0,0)
        print "Error fitting.symm_exponential_fit: Could not fit, parameters set to default"
        print str(e)
    return p, perr
# end symm_exponential_fit


'''
Fits data $x and $y to two exponential functions defined by fitting.symm_exp, one on each side
of the data as defined by the default parameters, centered by default.

Returns the fit parameters and the errors in the fit parameters for both left and right fits
as (pl, plerr, pr, prerr)

Default parameters:

$p0 is the starting fit parameters,
leave as-1 to estimate starting parameters from data

$xl is the index where the data for the left fit ends, centered by default
i.e. the left fit goes from xstart to xl

$xr is the index where the data for the right fit begins, centered by default
i.e. the right fit goes from xr to xstop

$xstart is the first data point to include in the fit,
leave as -1 to start with the first element in $x and $y

$xstop is the last data point to include in the fit,
leave as -1 to start with the last element in $x and $y
'''
def double_exponential_fit(x, y, p0=-1, xl=-1, xr=-1, xstart=-1, xstop=-1):
    l = len(y)
    if len(x) != l :
        print "Error fitting.symm_exponential_fit: X and Y data must have the same length"
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        a = np.mean(y)
        b = np.mean(y[9*l/20:11*l/20]) - a
        t = (x[xstop-1]-x[xstart] )/4.0
        p0=(a, b, t)
    if xl == -1:
        xl = l/2 # -20
    if xr == -1:
        xr = l/2 # -20

    try:
        pl, plconv = fit(symm_exp, t[xstart:xl], diff[xstart:xl], p0=p0)
        plerr = np.sqrt(np.diag(plconv))
    except Exception as e:
        pl = p0
        plerr = (0,0,0)
        print "Error fitting.double_exponential_fit: Could not fit, parameters set to default"
        print str(e)

    try:
        pr, prconv = fit(symm_exp, t[xr:xstop], diff[xr:xstop], p0=p0)
        prerr = np.sqrt(np.diag(prconv))
        #pr = p0
    except Exception as e:
        pr = p0
        prerr = (0,0,0)
        print "Error fitting.double_exponential_fit: Could not fit, parameters set to default"
        print str(e)
    return pl, plerr, pr, prerr
''

'''
A Power Law Function

y = A + B*x^g
'''
def power_law(x,A,B,g):
    return A + B*np.power(x,g)
# end power_law

'''
Fits data $x and $y to a symmetric exponential function defined by fitting.power_law

Returns the fit parameters and the errors in the fit parameters as (p, perr)

Default parameters:

$p0 is the starting fit parameters,
leave as-1 to estimate starting parameters from data

$xstart is the first data point to include in the fit,
leave as -1 to start with the first element in $x and $y

$xstop is the last data point to include in the fit,
leave as -1 to start with the last element in $x and $y
'''
def power_law_fit(x, y, p0=-1, xstart=-1, xstop=-1, p_default=None, perr_default=None):
    l = len(y)
    if len(x) != l :
        print "Error fitting.symm_exponential_fit: X and Y data must have the same length"
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        b, a = np.polyfit(x,y,1.0)
        g = 1.0
        p0=(a, b, g)
    try:
        p, plconv = fit(power_law, x[xstart:xstop], y[xstart:xstop], p0=p0, maxfev=2000)
        perr = np.sqrt(np.abs(np.diag(plconv)))
    except Exception as e:
        if p_default is None:
            p = p0
            perr = (0,0,0)
            print "Error fitting.symm_exponential_fit: Could not fit, parameters set to default"
            print str(e)
        else:
            p = p_default
            if perr_default is None:
                perr = (0,0,0)
            else:
                perr = perr_default
    return p, perr
# end power_law_fit

'''
A generic lowpass filter

$data is the data to be lowpassed, considered to be sampled at 1 Hz

$cutoff is the cutoff frequency in units of the nyquist frequency, must be less than 1

$samprate is the smaple rate in Hz

'''
def lowpass(data, cutoff=0.05, samprate=1.0):
    b,a = butter(2,cutoff/(samprate/2.0),btype='low',analog=0,output='ba')
    data_f = filtfilt(b,a,data)
    return data_f
# end lowpass

'''
Returns a normalized fast fourier transform of the given data
'''
def normfft(d):
    n = len(d)
    f = fft(d)
    f = 2.0*np.abs(f)/n
    return f
# end normfft
