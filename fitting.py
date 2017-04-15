'''
fitting.py

A module for fitting data

Last updated February 2016

by Trevor Arp

'''
import numpy as np
import scipy as sp
import warnings

from scipy.optimize import curve_fit as fit
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq

'''
Symmetric Exponential Function

y = A + B*Exp(-|x-t0/tau|)
'''
def symm_exp(x,A,B,tau, t0):
    return A + B*np.exp(-np.abs((x-t0)/tau))
# end symm_exp

'''
A Power Law Function

y = A*x^g + I0
'''
def power_law(x, A, g, I0):
    return A*np.power(x,g) + I0
# end power_law

'''
A one dimensioal Gaussian function given by

f(x,y) = A*Exp[ -(x-x0)^2/(2*sigma^2)]
'''
def guass(x, A, x0, sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))
# end guass2D

'''
A two dimensional Gaussian function given by

f(x,y) = A* Exp[ -(x-x0)^2/(2*sigmax^2) - (y-y0)^2/(2*sigmay^2) ]

where X is the corrdinate containing both x and y where x = X[0], y = X[1]
'''
def guass2D(X, A, x0, sigmax, y0, sigmay):
    return A*np.exp(-(X[0]-x0)**2/(2*sigmax**2) - (X[1]-y0)**2/(2*sigmay**2))
# end guass2D

'''
A symmetric bi-exponential function with a fast and slow component

y = A + B*Exp(-|x/tau_slow|) + C*Exp(-|x/tau_fast|)
'''
def biexponential(x, A, B, tauS, C, tauF, t0):
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF))
# end biexponential


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
def symm_exponential_fit(x, y, p0=-1, xstart=-1, xstop=-1, p_default=None, perr_default=None):
    l = len(y)
    if len(x) != l :
        print("Error fitting.symm_exponential_fit: X and Y data must have the same length")
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        a = np.mean(y)
        b = np.mean(y[int(9*l/20):int(11*l/20)]) - a
        t = (x[xstop-1]-x[xstart] )/4.0
        p0=(a, b, t, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = fit(symm_exp, x[xstart:xstop], y[xstart:xstop],p0=p0)
            perr = np.sqrt(np.diag(plconv))
    except Exception as e:
        if p_default is None:
            p = p0
            perr = (0,0,0,0)
            #print("Error fitting.symm_exponential_fit: Could not fit, parameters set to default")
            #print(str(e))
        else:
            p = p_default
            if perr_default is None:
                perr = (0,0,0,0)
            else:
                perr = perr_default
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
        print("Error fitting.symm_exponential_fit: X and Y data must have the same length")
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pl, plconv = fit(symm_exp, t[xstart:xl], diff[xstart:xl], p0=p0)
            plerr = np.sqrt(np.diag(plconv))
    except Exception as e:
        pl = p0
        plerr = (0,0,0)
        print("Error fitting.double_exponential_fit: Could not fit, parameters set to default")
        print(str(e))

    try:
        pr, prconv = fit(symm_exp, t[xr:xstop], diff[xr:xstop], p0=p0)
        prerr = np.sqrt(np.diag(prconv))
        #pr = p0
    except Exception as e:
        pr = p0
        prerr = (0,0,0)
        print("Error fitting.double_exponential_fit: Could not fit, parameters set to default")
        print(str(e))
    return pl, plerr, pr, prerr
# end double_exponential_fit

'''
A symmetric bi-exponential function with a fast and slow component. With a penalization parameter
to guarrentee that the fast and slow components are the correct order. Only needs to be used
for the fit

y = A + B*Exp(-|x/tau_slow|) + C*Exp(-|x/tau_fast|)
'''
def biexponential_pen(x, A, B, tauS, C, tauF, t0):
    if 1.5*tauF > tauS:
        penalization = 10000.0*tauF
    else:
        penalization = 0.0
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF)) + penalization
# end biexponential

'''
Fits data $x and $y to a biexponential function defined by fitting.biexponential

Returns the fit parameters and the errors in the fit parameters as (p, perr)

Default parameters:

$p0 is the starting fit parameters,
leave as-1 to estimate starting parameters from data

$xstart is the first data point to include in the fit,
leave as -1 to start with the first element in $x and $y

$xstop is the last data point to include in the fit,
leave as -1 to start with the last element in $x and $y
'''
def biexponential_fit(x, y, p0=-1, xstart=-1, xstop=-1, p_default=None, perr_default=None):
    l = len(y)
    if len(x) != l :
        print("Error fitting.symm_exponential_fit: X and Y data must have the same length")
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        ts_start = np.abs(x[0] - x[l-1])/2.0
        tf_start = ts_start/50.0
        p0 = (np.min(y), np.max(y)/10.0, ts_start, np.max(y), tf_start, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = fit(biexponential_pen, x, y, p0=p0)
            #p, plconv = fit(biexponential, x, y, p0=p0)
            perr = np.sqrt(np.diag(plconv))
    except Exception as e:
        #print('Error: Could not fit')
        if p_default is None:
            p = p0
            perr = (0,0,0,0,0,0)
        else:
            p = p_default
            if perr_default is None:
                perr = (0,0,0,0,0,0)
            else:
                perr = perr_default
    return p, perr
# end biexponential_fit

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
        print("Error fitting.symm_exponential_fit: X and Y data must have the same length")
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        b, a = np.polyfit(x,y,1.0)
        g = 1.0
        p0=(b, g, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = fit(power_law, x[xstart:xstop], y[xstart:xstop], p0=p0, maxfev=2000)
            perr = np.sqrt(np.abs(np.diag(plconv)))
    except Exception as e:
        #print(str(e)) #debug
        if p_default is None:
            p = p0
            perr = (0,0,0)
            print("Error fitting.power_law_fit: Could not fit, parameters set to default")
            print(str(e))
        else:
            p = p_default
            if perr_default is None:
                perr = (0,0,0)
            else:
                perr = perr_default
    return p, perr
# end power_law_fit

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
def guass_fit(x, y, p0=-1, xstart=-1, xstop=-1, p_default=None, perr_default=None):
    l = len(y)
    if len(x) != l :
        print("Error fitting.guass_fit: X and Y data must have the same length")
        return
    if xstart == -1:
        xstart = 0
    if xstop == -1:
        xstop = l
    if p0 == -1:
        a = np.max(y)
        sigma = 1.0
        x0 = x[int(len(x)/2)]
        p0=(a, sigma, x0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = fit(guass, x[xstart:xstop], y[xstart:xstop], p0=p0)
            perr = np.sqrt(np.diag(plconv))
    except Exception as e:
        if p_default is None:
            p = p0
            perr = (0,0,0)
            #print("Error fitting.symm_exponential_fit: Could not fit, parameters set to default")
            #print(str(e))
        else:
            p = p_default
            if perr_default is None:
                perr = (0,0,0)
            else:
                perr = perr_default
    return p, perr
    #return p0
# end guass_fit


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
def lp_scan_cols(datacube, cutoff=0.05, samprate=1.0):
    rows, cols = datacube.shape
    original = np.copy(datacube)
    for i in range(cols):
        datacube[:,i] = lowpass(original[:,i], cutoff=cutoff, samprate=samprate)
    return datacube
# end lp_cube_cols

'''
Returns a normalized fast fourier transform of the given data
'''
def normfft(d):
    n = len(d)
    f = fft(d)
    f = 2.0*np.abs(f)/n
    return f
# end normfft

'''
Takes a two dimensional array of data $data and fits it to some function $func where func is
define such that data = func(X, *args) where x = X[0,:], y = X[1,:], the parameters $x and $y
define the ranges over which the coordinates range for data, i.e. $data = func(($x, $y), *p0)

$p0 is the initial parameters

returns the parameters array and covariance matrix output from scipy.optimize.curve_fit
'''
def fit_2D(func, x, y, data, p0):
    M = len(x)
    N = len(y)
    if (N,M) != data.shape:
        print("Error fit_2D: $x and $y must have lengths N and M for an MxN data array")
        raise ValueError
    coords = np.zeros((2, N*M))
    x_m = np.zeros((N,M))
    y_m = np.zeros((N,M))
    for i in range(M):
        x_m[i,:] = x
    for i in range(N):
        y_m[:,i] = y
    coords[0] = x_m.flatten()
    coords[1] = y_m.flatten()
    return fit(func, coords, data.flatten(), p0=p0)
# end fit_2D


'''
Takes two input images $d1 and $d2 and computes the spatial shift between them (assuming they are
similar images)

$d1 and $d2 are the two images to compute the shift between. Assumes they are the same size

When computing the shift the fitting will consider the region +/- 1/$frac around the center of the
autocorrelation, a larger fraction will take more computational time.

$debugAC when true will return the autocorrelation along with the shift

'''
def compute_shift(d1, d2, frac=5.0, debugAC=False):
    rows, cols = d1.shape
    ac = sp.signal.fftconvolve(d1, d2[::-1, ::-1]) #sp.signal.correlate2d(d1, d2)
    mx = np.unravel_index(ac.argmax(), ac.shape)
    N, M = ac.shape
    x = np.linspace(0, M, M)
    y = np.linspace(0, N, N)
    l = int(rows/frac)
    Z = ac[N-l:N+l, M-l:M+l]
    try:
        p, pcorr = fit_2D(guass2D, x[M-l:M+l], y[N-l:N+l], Z, (np.max(ac), mx[1], 1.0, mx[0], 1.0))
    except Exception as e:
        print("Error fitting.exact_arg_max: Could not fit autocorrlation to a guassian")
        print(str(e))
        raise
    sft = (-p[3]+rows-1, -p[1]+cols-1)
    if debugAC:
        return sft, ac
    else:
        return sft
# end compute_shift

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
