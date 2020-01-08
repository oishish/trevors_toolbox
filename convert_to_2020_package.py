'''
In January of 2020 I converted my toolbox from a series of modules into a properly structured
package, scripts written prior to that may not be backwards comparable. This script should make
(most of) the nessecary changes to convert a script to the new 2020 format.

Run this script with the directory in question as a command lin eargument, will recursivly search
the directory in question and convert any python files it finds.

BACKUP BEFORE USING THIS SCRIPT!
'''

from sys import argv
from os.path import exists, join
from os import walk
from traceback import format_exc

# Old modules names, switch to 'tb.module'
modules = ['utils', 'scans', 'display', 'visual', 'calibration', 'fitting', 'sim', 'postprocess', 'process']

# Old functions that are still availible in the new format, switch to 'tb.function'
functions = ['find_run', 'find_savefile', 'slowscan_2_line', 'range_from_log', 'load_run', 'get_processed_data', 'ProcessRun', 'dataimg']

# Old functions that are no longer availible in the new format, switch to 'tb.module.function'
# dictionary where the key is the old keyword and the value is the location in the module
legacy_functions = {"get_figure":"display.get_figure", "get_viridis":"display.get_viridis", "format_plot_axes":"display.format_plot_axes", "set_img_ticks":"display.set_img_ticks", "scale_bar_plot":"display.scale_bar_plot"}
legacy = list(legacy_functions.keys())

# List of all keywords
keywords = modules + functions + legacy

def convert_to_2020(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    #

    if len([i for i in lines if "from toolbox import *" in i]) > 0:
        for i in range(len(lines)):
            if any(x in lines[i] for x in keywords):
                if "import" in lines[i]: # Lines with import statements, only convert keywords from modules
                    for module in modules:
                        if "from "+module in lines[i]:
                            lines[i] = lines[i].replace(module, "toolbox."+module)
                else: # non-import lines with keywords get converted, try to ensure they are not in comments or other names
                    go = True # ensures that if nothing is done to a line, subsequent items are still checked for
                    if any(x in lines[i] for x in modules): # first search for modules
                        for module in modules:
                            if go and module+'.' in lines[i]:
                                lines[i] = lines[i].replace(module+'.', "tb."+module+'.')
                                go = False
                    if go and any(x in lines[i] for x in functions): # then functions
                        for func in functions:
                            if func+'(' in lines[i]:
                                lines[i] = lines[i].replace(func+'(', "tb."+func+'(')
                                go = False
                    elif go and any(x in lines[i] for x in legacy): # then legacy functions
                        for func in legacy:
                            if func+'(' in lines[i]:
                                lines[i] = lines[i].replace(func+'(', "tb."+legacy_functions[func]+'(')
                                go = False
            elif lines[i] == "input()\n": # A holdover from when I used python 2, using input by itself to block execution
                lines[i] = "plt.show()\n"
        # end for

        # Insert in standard peices in place of wild import, so this after changing the lines
        # as it involves insertions
        for i in range(len(lines)):
            if "from toolbox import *" in lines[i]:
                lines[i] = "import toolbox as tb\n"
                lines.insert(i, "import matplotlib.pyplot as plt\n")
                lines.insert(i, "import numpy as np\n")
                if len([i for i in lines if "sp." in i]) > 0:
                    lines.insert(i, "import scipy as sp\n")
                break
        # end for

        # Write the converted file
        with open(filename, 'w') as f: # the +'a' is for debugging
            f.writelines(lines)
        return 0
    else:
        return -1
#

# Search for all the python files in a directory tree, return paths as a list
def search_files(directory='.', extension='.py'):
    fls = []
    extension = extension.lower()
    for dirpath, dirnames, files in walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                fls.append(join(dirpath, name))
    return fls
#

if __name__ == "__main__":
    if len(argv) < 2:
        print("Error: script requires arguments")
    if exists(argv[1]):
        fls = search_files(argv[1])
        cnt = 0
        for f in fls:
            try:
                r = convert_to_2020(f)
            except:
                print(f)
                print(format_exc())
            if r == 0: # Successfully converted
                cnt += 1
                print("converted " + str(f))
        print("")
        print(str(cnt) + ' files converted')
    else:
        raise ValueError("File not found: " + str(argv[1]))
#
