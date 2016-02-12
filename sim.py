'''
sim.py

A module for simulating various things

Last updated January 2016

by Trevor Arp

'''
import numpy as np
import scipy as sp
from multiprocessing import Pool
from timeit import default_timer as timer
import datetime
import traceback

'''
Heaviside Theta function
'''
def HTheta(x):
    return 0.5*(np.sign(x) + 1)
#

'''
Boxcar function

Composed of two HTheta function that are 1.0 from start to stop and is zero everywhere else
'''
def Box(x, start, stop):
    return HTheta(x-start) - HTheta(x-stop)
#


'''
Multipurpose parallel processor

Takes a function and an array of arguments, evaluates the function with the given arguments for each point,
processing in parallel using $ncores number of parallel processes. Returns the results as a numpy ndarray

$args_array is the array of arguments to the input function $func. $func can only accept one argement,
but it can be a list or tuple

Will display progress if $display is True

'''
def multiprocess2D(func, args_array, ncores=4, display=True):
    pool = Pool(processes=ncores)
    rows = len(args_array)
    cols = len(args_array[0])
    output = np.zeros((rows, cols))
    if rows > 10:
        disp_rows = np.arange(rows/10, rows, rows/10)
    else:
        disp_rows = np.arange(1, rows, 1)
    if display:
        print "Parallel Processing Started with " + str(ncores) + " subprocesses"
    t0 = timer()
    for i in range(rows):
        worker_args = []
        for j in range(cols):
            worker_args.append(args_array[i][j])
        try:
            out = pool.map(func, worker_args)
            for j in range(cols):
                output[i,j] = out[j]
            if display and i in disp_rows:
                print str(round(100*i/float(rows))) + "% Complete"
        except Exception as e:
            print "Exception in sim.multiprocessing2D: Cannot Process"
            print traceback.print_exc()
            print "sim.multiprocessing2D: Exiting Process"
            break
    tf = timer()
    if display:
        print " "
        dt = tf-t0
        print "Computations Completed in: " + str(datetime.timedelta(seconds=dt))
    return output
# end multiprocess2D
