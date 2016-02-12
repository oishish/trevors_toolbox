'''
postprocess.py

A module for general post-processing of data, i.e. general functions relating to filtering, image
processing, etc. that work with processed data, not raw data and is not fitting (see fitting.py
for that).

Last updated February 2016

by Trevor Arp
'''
from toolbox import *

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
