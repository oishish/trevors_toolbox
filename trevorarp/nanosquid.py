'''
A module for nanosquid specific code.
'''
import numpy as np

def cal_attocube_to_um(x=None, y=None, z=None, xmax=40, ymax=40, zmax=20, vmax=7.5, centerxy=True):
    '''
    Calibrate the voltage sent to the attocubes into position in microns based on the extension
    at maximum voltage ouput.

    Args:
        x : The X axis values to calibrate
        y : The Y axis values to calibrate
        z : The Z axis values to calibrate
        xmax : The X axis maximum extension
        ymax : The Y axis maximum extension
        vmax : The voltage to apply for maximum extension
        cetnerxy: if True will place (0,0) in center of voltage range.
    '''
    ret = []
    if x is not None:
        x = xmax*x/vmax
        if centerxy:
            x = x - xmax/2
        ret.append(x)
    if y is not None:
        y = ymax*y/vmax
        if centerxy:
            y = y - ymax/2
        ret.append(y)
    if z is not None:
        z = zmax*z/vmax
        ret.append(z)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
#
