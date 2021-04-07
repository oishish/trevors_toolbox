'''
A Module for data processing between from various formats
'''

import h5py
from os.path import exists
import numpy as np

def datavault2numpy(filename):
    '''
    A tool to convert a datavault hdf5 file into python numpy

    Args:
        filename (str) : The path to the datavault file, if .hdf5 extention is not included it will
            add it before attempting to load.
    '''
    ext_test = filename.split('.')
    if len(ext_test) > 1 and ext_test[len(ext_test)-1] == 'hdf5':
        pass
    else:
        filename = filename + '.hdf5'
    if not exists(filename):
        raise IOError("File " + str(filename) + " not found.")
    f = h5py.File(filename)
    dv = f['DataVault']
    d = dv[...].tolist()
    return np.array(d)
#
