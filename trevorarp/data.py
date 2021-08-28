'''
A module for data processing between from various formats
'''

import h5py
import labrad

from os.path import exists
import numpy as np

def get_dv_data(identifier, remote=None, subfolder=None):
    '''
    A function to retreive data from the datavault using a nanosquid identifier and return is as numpy arrays

    Args:
        identifier (str): The specific
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
    '''
    if remote is None:
        cxn = labrad.connect()
    else:
        cxn = labrad.connect(remote, password='pass')

    dv = cxn.data_vault
    if subfolder is not None:
        dv.cd(subfolder)

    drs, fls = dv.dir()
    filename = [x for x in fls if identifier in x]

    if len(filename) == 0:
        raise IOError("Identifier " + identifier + " not found on this data vault.")
    elif len(filename) > 1:
        print("Warning files with duplicate identifiers detected, only the first one was retreived")
    datafile = filename[0]
    dv.open(datafile)
    data = np.array(dv.get())
    return data

def retrievefromvault(vaultdir, filename, host='localhost', password='pass'):
    '''
    A generic tool to retrieve files from a LabRAD datavault

    Args:
        vaultdir (str) : The sub directory of the vault ot find the files in (neglecting the .dir extension)
        filename (str) : The name of the file, neglecting the leading numbers or file extenstion, for
            example "data1" for the file "00001 - data1.hdf" if there are files with the same name but
            different numbers it will always retreive the first instance.
        host (str) : The host for the labrad connection, localhost by default.
        password (str) : The password for the labrad connection, localhost password by default.
    '''
    dv = labrad.connect('localhost', password='pass').data_vault
    for dir in vaultdir.split('\\'):
        dv.cd(dir)
    rt, fls = dv.dir()
    for fl in fls:
        if fl.split(' - ',1)[1] == filename:
            datafile = fl
            break
    dv.open(datafile)
    return np.array(dv.get())
#

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
