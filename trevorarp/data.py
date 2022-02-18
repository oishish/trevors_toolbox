'''
A module for data processing between from various formats
'''

import h5py
import labrad

from os.path import exists
import numpy as np

'''
nSOTColumnSpec allows generic nSOT data of particular types, corresponding to a specific filename
to be read in and unwrapped automatically.

Format is
{
"Name":[trace/retrace_index, 
(row_axis_index, row_axis_values, row_label),
(col_axis_index, col_axis_values), 
(dependent_1, ..., dependent_N), (dependent_1_label, ..., dependent_N_label)]}
'''
nSOTColumnSpec = {
## "nSOT vs. Bias Voltage and Field", ['Trace Index', 'B Field Index','Bias Voltage Index','B Field','Bias Voltage'],['DC SSAA Output','Noise']
"nSOT vs. Bias Voltage and Field":(0, (1,3,"B Field (T)"), (2,4,"SQUID Bias (V)"), (5,6), ("Feedback (V)", "Noise"))
}

def get_dv_data(identifier, remote=None, subfolder=None, retfilename=False):
    '''
    A function to retreive data from the datavault using a nanosquid identifier and return is as numpy arrays

    Args:
        identifier (str): The specific
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
        retfilename : If True will return the name of the datavault file along with the data
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
    if retfilename:
        return data, datafile
    else:
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

def get_reshaped_nSOT_data(iden, remote=None, subfolder=None):
    '''
    Gets a data set of a known nSOT measurement type and unwraps it from columns into a useful
    dataset based on a known format. Assumes that it has a column that is the index for the fast
    and slow axes along with their values.
    
    Args:
        iden (str) : The Unique identifier generated by datavault for an nSOT system.
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
    
    Returns in the format:
    row_values, colum_values, dependent_variables_trace, dependent_variables_retrace, labels
    Where dependent variables trace and retrace are in the order of the data vault and labels contains:
    (row_label, column_label, dependent_1_label, ..., dependent_N_label)
    
    '''
    d, fname = get_dv_data(iden, remote=remote, subfolder=subfolder, retfilename=True)
    
    sweeptype = fname.split(' - ')[2]
    if sweeptype in nSOTColumnSpec:
        trix, cvars, rvars, dvars, dvars_labels = nSOTColumnSpec[sweeptype]
    else:
        raise ValueError("Unique Identifier does not correspond to a known 2D data type in nSOTColumnSpec")
    
    try:
        trace = d[d[:,trix]==0,:]
        retrace = d[d[:,trix]==1,:]
        
        rows = int(np.max(trace[:,rvars[0]])) + 1
        cols = int(np.max(trace[:,cvars[0]])) + 1
        
        rvalues = np.reshape(trace[:,rvars[1]],(rows, cols), order='F')
        cvalues = np.reshape(trace[:,cvars[1]],(rows, cols), order='F')
    except ValueError:
        print("Error reshaping the data array, check that the specification is correct.")
        print(format_exc())
        return
    
    dependent = []
    dependent_retrace = []
    for ix in dvars:
        dependent.append(np.reshape(trace[:,ix],(rows, cols), order='F'))
        dependent_retrace.append(np.reshape(trace[:,ix],(rows, cols), order='F'))
    
    labels = (rvars[2], cvars[2], *dvars_labels)
    
    return rvalues, cvalues, dependent, dependent_retrace, labels
