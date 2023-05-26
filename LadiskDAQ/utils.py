"""
This files is meant for functions that are useful for setting up workflow and analysis of the measurements.
"""

import os
import numpy as np
import pickle

# open measurements:
def load_measurement(name: str, directory: str = ''):
    """
    Loads a measurement from a pickle file.
    """
    file_path = os.path.join(directory, name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def load_measurement_multiple_files(directory: str = '', contains: str = ''):
    """
    Loads all measurements with name mathcing the string and combines them into one measurement
    dictionary.
    """
    files = os.listdir(directory)
    files = [file for file in files if contains in file]
    measurement = {}
    for file in files:
        meas = load_measurement(os.path.join(directory , file))
        for source in meas.keys():
            if source not in measurement.keys():
                measurement[source] = meas[source]
            else: # we need to concatenate
                for key in meas[source].keys():
                    vals = meas[source][key]
                    if key == "video": # videos are saved as arrays in list
                        measurement[source][key] = [ np.concatenate((array, arr), axis=0) for array, arr 
                                                    in zip(measurement[source][key], meas[source][key]) ]
                        
                    if key == "data" or key == "time": # data is saved 2D array
                        measurement[source][key] = np.concatenate((measurement[source][key], meas[source][key]), axis=0)
                        
    return measurement