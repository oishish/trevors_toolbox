'''
postprocess.py

A module for general post-processing of data, i.e. general functions relating to filtering, image
processing, etc. that work with processed data, not raw data and is not fitting (see fitting.py
for that).

Last updated February 2016

by Trevor Arp
'''
import numpy as np
import scipy as sp
import fitting

from scipy import ndimage as ndi
from skimage import filters
from skimage.morphology import skeletonize, remove_small_objects

'''
Average fit error
chi =  sum( (obs-expected)^2/expected^2 )
'''
def avg_error(obs, expected):
    N = len(obs)
    s = 0.0
    for i in range(N):
        s += ((obs[i] - expected[i]) / expected[i] )**2
    s = np.sqrt(s/N)
    return s
# avg_error

'''
For a power data cube, filters the points based on some criteria

$d is the raw data and $power is the power parameter

$fit is the calculated power law fit

kwargs:

$fill is the gamma value to fill the points that are filtered out, default (None) fills with np.nan

Returns filtered values of $gamma and $Amplitude
'''
def filter_power_cube(d, power, fit, fill=None):
    rows, cols, N = d.shape
    if fill is None:
        fill = np.nan
    gamma = fit[:,:,1]
    params = fit[:,:,0:3]
    perr = fit[:,:,3:6]
    for i in range(rows):
        for j in range(cols):
            for k in range(2):
                if np.abs(perr[i,j,k]) > np.abs(params[i,j,k]):
                    gamma[i,j] = 0.0
    return gamma
# end filter_power_cube


'''
Filters a Space-Delay cube, similar to filter_power_cube

$t is the time delay, and fit is the fit to a symmetric exponential

kwargs:

$fill is the tau value to fill the points that are filtered out, default (None) fills with np.nan

$min_A is the minimum amplitude of the fit function, points less that this are filtered out

$max_tau is the maximum acceptable value of tau, points above this are filtered out

$max_terr is the maximum acceptable error in tau, points above this are filtered out

returns filtered tau values

'''
def filter_delay_cube(t, fit,
    fill=None,
    min_A=0.1,
    max_tau=100,
    max_terr=5.0
    ):
    if fill is None:
        fill = np.nan
    rows, cols, N = np.shape(fit)
    tau = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.abs(fit[i,j,1])<min_A or np.abs(fit[i,j,6])>max_terr or np.abs(fit[i,j,2])>max_tau:
                tau[i,j] = fill
            else:
                tau[i,j] = np.abs(fit[i,j,2])
    return tau
# end filter_delay_cube

'''
Takes a fit matrix (from a power data cube) and filters the gamma values (gamma = fit[:,:,2]) based on the
sign of the slope (slovbe B = fit[:,:,1]). Returns two matirices, with values of positive slope
and one with values of negative slope.

$mingamma is the minimum absolute value of gamma needed for a point to be considered, if if
doesn't meet this threshold it will not be in either returned matrix.

If $nanfill is true will fill the empty points in both arrays with NaNs for transparent plotting
'''
def filter_slopes_power_cube(fit, mingamma=0.0, nanfill=False):
    g = fit[:,:,1]
    B = fit[:,:,0]
    rows, cols, dim = fit.shape
    if dim != 4:
        s = "Error filter_slopes_power_cube: input fit cube must have dimension of 6, based on the"
        s += "standard power law fit."
    if nanfill:
        g_above = np.empty((rows,cols))
        g_below = np.empty((rows,cols))
        g_above[:,:] = np.nan
        g_below[:,:] = np.nan
    else:
        g_above = np.zeros((rows,cols))
        g_below = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if np.abs(g[i,j]) > mingamma:
                if B[i,j] > 0.0:
                    g_above[i,j] = g[i,j]
                else:
                    g_below[i,j] = g[i,j]
    return g_above, g_below
# end filter_slopes_power_cube

'''
Finds sharp edges in the input image $d using a sobel filter, and morphological operations

if $remove_small is true then small domains will be removed from the filtered image prior to the final
calculation of the edge, has the potential to remove some of the edge
'''
def find_sharp_edges(d, remove_small=False):
    # edge filter
    edge = filters.sobel(d)

    # Convert to binary image
    thresh = filters.threshold_li(edge)
    edge = edge > thresh

    # Close the gaps
    edge = ndi.morphology.binary_closing(edge)

    # If desiered remove small domains
    if remove_small:
        edge = remove_small_objects(edge)

    # Skeletonize the image down to minimally sized features
    edge = skeletonize(edge)
    return edge
# end find_sharp_edges
