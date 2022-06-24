'''
A module for data processing between from various formats
'''

import h5py
import labrad

from os.path import exists
import numpy as np
from traceback import format_exc

'''
nSOTColumnSpec allows generic nSOT data of particular types, corresponding to a specific filename
to be read in and unwrapped automatically.

Format is
{
"Name":[trace/retrace_index, reshape_order,
(column_axis_index, column_axis_values, column_label),
(row_axis_index, row_axis_values, row_labels), 
(dependent_1, ..., dependent_N), (dependent_1_label, ..., dependent_N_label)]}

reshape_type is the "order" parameter to pass to reshape that determines how the elements are read
out. Options are "C", "F", and "A" see the numpy.reshape documentation. Generally "F" means the slow
axis is the first index and the fast index is the second.
"C" means the fast axis is the second axis.

trace/retrace_index should be negative if there is no such index

The dependent variables (and labels) can be ("*", ix) where all data columns starting with ix onwards
are assumed to be unspecified independent Variables. In which case the labels parameter (though it 
should be present) is ignored and labels will be automatically generated.
'''
nSOTColumnSpec = {
# "nSOT vs. Bias Voltage and Field", ['Trace Index', 'B Field Index','Bias Voltage Index','B Field','Bias Voltage'],['DC SSAA Output','Noise']
"nSOT vs. Bias Voltage and Field":(0, "F", (1,3,"B Field (T)"), (2,4,"SQUID Bias (V)"), (5,6), ("Feedback (V)", "Noise")),
# "nSOT Scan Data " + self.fileName, ['Retrace Index','X Pos. Index','Y Pos. Index','X Pos. Voltage', 'Y Pos. Voltage'],in_name_list
"nSOT Scan Data unnamed":(0, "C", (1,3,"X Voltage"),(2,4,"Y Voltage"),('*',5),('*')),
# 'FourTerminal MagneticField ' + self.Device_Name, ['Magnetic Field index', 'Gate Voltage index', 'Magnetic Field', 'Gate Voltage'],["Voltage", "Current", "Resistance", "Conductance"]
"FourTerminal MagneticField Device Name":(-1, "C", (1,3,"Gate Voltage"), (0,2,"B Field"),(4,5,6,7), ("Voltage", "Current", "Resistance", "Conductance")),
# "Four Terminal Landau Voltage Biased", ['Gate Voltage index', 'Magnetic Field index',"Gate Voltage", "Magnetic Field"], ["Voltage Lock-In", "Current Lock-In"]
"Four Terminal Voltage Biased":(-1, "C", (0,2,"Gate Voltage"), (1,3,"B Field"),(4,5), ("Voltage", "Current")),
# "Dual Gate Voltage Biased Transport", ["p0 Index", "n0 Index","p0", "n0"], ["Vt", "Vb", "Voltage Lock-In", "Current Lock-In"]
"Dual Gate Voltage Biased Transport":(-1, "F", (0,2,"p0"), (1,3,"n0"), (4,5,6,7), ("Vt", "Vb", "Voltage", "Current")),
}

def get_dv_data(identifier, remote=None, subfolder=None, params=False, retfilename=False):
    '''
    A function to retreive data from the datavault using a nanosquid identifier and return is as numpy arrays

    Args:
        identifier (str): The specific
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
        params (bool) : If True will return any parameters from the data vault file.
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
    filename = [x for x in fls if identifier+" " in x] # the space prevents finding multiples of ten, for example iden-1 and iden-10

    if len(filename) == 0:
        raise IOError("Identifier " + identifier + " not found on this data vault.")
    elif len(filename) > 1:
        print("Warning files with duplicate identifiers detected, only the first one was retreived")
        print(filename)
    datafile = filename[0]
    dv.open(datafile)
    data = np.array(dv.get())
    
    plist = dv.get_parameters()
    parameters = dict()
    if plist is not None:
        for p in plist:
            if isfloat(p[1]):
                parameters[p[0]] = float(p[1])
            else:
                parameters[p[0]] = p[1]
    
    if retfilename and params:
        return data, parameters, datafile
    elif retfilename:
        return data, datafile
    elif params:
        return data, parameters
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

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_reshaped_nSOT_data(iden, remote=None, subfolder=None, params=False):
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
        params (bool) : If True will return any parameters from the data vault file.
    
    Returns in the format:
    row_values, colum_values, dependent_variables_trace, dependent_variables_retrace, labels
    Where dependent variables trace and retrace are in the order of the data vault and labels contains:
    (row_label, column_label, dependent_1_label, ..., dependent_N_label). If there is not distriction
    between trace and retrace then dependent_variables_trace and dependent_variables_retrace will be the same.
    
    '''
    d, dvparams, fname = get_dv_data(iden, remote=remote, subfolder=subfolder, retfilename=True, params=True)
    
    sweeptype = fname.split(' - ')[2]
    if sweeptype in nSOTColumnSpec:
        trix, order, cvars, rvars, dvars, dvars_labels = nSOTColumnSpec[sweeptype]
    else:
        raise ValueError("Unique Identifier does not correspond to a known 2D data type in nSOTColumnSpec")
    
    try:
        if trix >= 0:
            trace = d[d[:,trix]==0,:]
            retrace = d[d[:,trix]==1,:]
        else:
            trace = d
            retrace = d
        
        rows = int(np.max(trace[:,rvars[0]])) + 1
        cols = int(np.max(trace[:,cvars[0]])) + 1
        
        rvalues = np.reshape(trace[:,rvars[1]],(rows, cols), order=order)
        cvalues = np.reshape(trace[:,cvars[1]],(rows, cols), order=order)
    except ValueError:
        print("Error reshaping the data array, check that the specification is correct.")
        print(format_exc())
        return
    
    dependent = []
    dependent_retrace = []
    if dvars[0] == "*":
        l, dcols = trace.shape
        dvars_labels = []
        for ix in range(dvars[1], dcols):
            tr = np.reshape(trace[:,ix],(rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
            dvars_labels.append("Column "+str(ix))
    else:
        for ix in dvars:
            tr = np.reshape(trace[:,ix],(rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
    labels = (rvars[2], cvars[2], *dvars_labels)
    
    if params:
        return rvalues, cvalues, dependent, dependent_retrace, labels, dvparams
    else:
        return rvalues, cvalues, dependent, dependent_retrace, labels
