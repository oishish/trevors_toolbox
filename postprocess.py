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

Average fit error
chi =  sum( (obs-expected)^2/expected^2 )
'''
def filter_power_cube(d, power, fit, max_chi=0.5):
    rows, cols, N = d.shape
    f = fitting.power_law
    gamma = fit[:,:,2]
    gamma_err = fit[:,:,5]
    A = fit[:,:,1]
    out_gamma = np.zeros((rows, cols))
    out_gamma_err = np.zeros((rows, cols))
    chi = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            expected = f(power[i,:], fit[i,j,0], fit[i,j,1], fit[i,j,2])
            chi[i,j] = avg_error(np.abs(d[i,j,:]), expected)
            if chi[i,j] > max_chi or gamma[i,j] <= 0.0:
                out_gamma[i,j] = None
                out_gamma_err[i,j] = None
            else:
                out_gamma[i,j] = gamma[i,j]
                out_gamma_err[i,j] = gamma_err[i,j]
    return out_gamma, out_gamma_err, chi
# end filter_power_cube

'''
Takes a fit matrix (from a power data cube) and filters the gamma values (gamma = fit[:,:,2]) based on the
sign of the slope (slovbe B = fit[:,:,1]). Returns two matirices, with values of positive slope
and one with values of negative slope.

$mingamma is the minimum absolute value of gamma needed for a point to be considered, if if
doesn't meet this threshold it will not be in either returned matrix.

If $nanfill is true will fill the empty points in both arrays with NaNs for transparent plotting
'''
def filter_slopes_power_cube(fit, mingamma=0.0, nanfill=False):
    g = fit[:,:,2]
    B = fit[:,:,1]
    rows, cols, dim = fit.shape
    if dim != 6:
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
