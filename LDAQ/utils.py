"""
This files is meant for functions that are useful for setting up workflow and analysis of the measurements.
"""

import os
import pickle

import numpy as np
from tqdm import tqdm


# open measurements:
def load_measurement(name: str, directory: str = ''):
    """
    Loads a measurement from a pickle file.
    
    Args:
        name (str): name of the measurement file
        directory (str): directory where the measurement file is located
        
    Returns:
        measurement (dict): dictionary containing the measurement
    """
    if directory == '':
        file_path = name
    else:
        file_path = os.path.join(directory, name)

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_measurement_multiple_files(directory: str = None,
                                    contains: str = '',
                                    pbar: bool = False):
    """
    Loads all measurements with name matching the string and combines them into one measurement
    dictionary.
    
    If data is too large to fit into memory, consider using load_measurement_multiple_files_memmap.
    
    Args:
        directory (str): directory where the measurement files are located
        contains (str): string that the measurement files should contain in their name
        pbar (bool, optional): If True, displays a progress bar using tqdm. Defaults to False.
        
    Returns:
        measurement (dict): dictionary containing concatenated measurement datra from multiple files.
    """
    files = os.listdir(directory)
    files = [file for file in files if contains in file]
    measurement = {}
    for file in files if not pbar else tqdm(files):
        if directory is None:
            meas = load_measurement(file)
        else:
            meas = load_measurement(os.path.join(directory, file))
        for source in meas.keys():
            if source not in measurement.keys():
                measurement[source] = meas[source]
            else:  # we need to concatenate
                for key in meas[source].keys():
                    vals = meas[source][key]
                    if key == "video":  # videos are saved as arrays in list
                        measurement[source][key] = [
                            np.concatenate(
                                (array, arr), axis=0) for array, arr in zip(
                                    measurement[source][key], meas[source][key])
                        ]

                    if key == "data" or key == "time":  # data is saved 2D array
                        measurement[source][key] = np.concatenate(
                            (measurement[source][key], meas[source][key]),
                            axis=0)

    return measurement


def load_measurement_multiple_files_memmap(directory: str,
                                           contains: str = '',
                                           pbar: bool = False,
                                           tmp_dir: str = "tmp"):
    """
    Loads and concatenates measurement data from multiple files using memory mapping.

    This function scans a directory for measurement files whose names contain a specified substring,
    loads their contents using `load_measurement`, and concatenates the data per source across files.
    The combined data is written to memory-mapped files stored in a temporary directory to handle 
    large datasets efficiently. Supports progress tracking via tqdm if `pbar=True`.

    Args:
        directory (str): Directory containing measurement files.
        contains (str, optional): Substring to filter relevant files by name. Defaults to ''.
        pbar (bool, optional): If True, displays a progress bar using tqdm. Defaults to False.
        tmp_dir (str, optional): Directory to store temporary memory-mapped files. Defaults to "tmp".

    Returns:
        dict: A dictionary where each key corresponds to a measurement source, and each value is a
              dictionary with the following keys (depending on availability in the source):
                  - 'data' (np.memmap): Memory-mapped array containing concatenated data.
                  - 'video' (list of np.memmap): Memory-mapped arrays for video channels.
                  - 'time' (np.memmap): Memory-mapped array containing time information.
                  - 'channel_names' (list): Names of data channels.
                  - 'channel_names_video' (list): Names of video channels.
                  - 'sample_rate' (float): Sampling rate of the measurement.
    """
    files = [f for f in os.listdir(directory) if contains in f]

    if not files:
        raise ValueError("No matching files found in directory.")

    file0 = files[0]
    meas0 = load_measurement(os.path.join(directory, file0))
    sources = meas0.keys()

    total_rows = {}
    sample_data_shape = {}
    sample_video_shape = {}

    for source in sources:
        total_rows[source] = 0
        if len(meas0[source]['channel_names']) > 0:
            sample_data_shape[source] = meas0[source]['data'].shape[1]
        if 'channel_names_video' in meas0[source]:
            sample_video_shape_ = {}
            for c_n_v in meas0[source]['channel_names_video']:
                sample_video_shape_[c_n_v] = meas0[source]['video'][0].shape[
                    -2:]

            sample_video_shape[source] = sample_video_shape_

    # Pass 1: Calculate total number of rows
    for file in files if not pbar else tqdm(files):
        meas = load_measurement(os.path.join(directory, file))
        for source in sources:
            if source in sample_data_shape:
                data = meas[source]['data']
                total_rows[source] += data.shape[0]
            elif source in sample_video_shape:
                video = meas[source]['video'][0]
                total_rows[source] += video.shape[0]
            else:
                raise ValueError('No data or video in measurement.')

    if not os.path.exists(tmp_dir):
        print(f'Creating temporary directory {tmp_dir}')
        os.makedirs(tmp_dir)

    # Allocate memmaps
    data_mmap = {}
    video_mmap = {}
    time_mmap = {}
    for source in sources:
        if source in sample_data_shape:
            data_mmap[source] = np.memmap(
                os.path.join(tmp_dir, f'{source}_data_mmap.dat'),
                dtype=meas0[source]['data'].dtype,
                mode='w+',
                shape=(total_rows[source], sample_data_shape[source]))
        if source in sample_video_shape:
            video_mmap_ = {}
            for i, c_n_v in enumerate(sample_video_shape[source]):
                video_mmap_[c_n_v] = np.memmap(
                    os.path.join(tmp_dir, f'{source}_{c_n_v}_video_mmap.dat'),
                    dtype=meas0[source]['video'][i].dtype,
                    mode='w+',
                    shape=(total_rows[source],
                           *sample_video_shape[source][c_n_v]))

            video_mmap[source] = video_mmap_

        time_mmap[source] = np.memmap(os.path.join(tmp_dir,
                                                   f'{source}_time_mmap.dat'),
                                      dtype=meas0[source]['time'].dtype,
                                      mode='w+',
                                      shape=(total_rows[source],))

    # Pass 2: Load and write to memmap
    offset = {source: 0 for source in sources}
    for file in files if not pbar else tqdm(files):
        meas = load_measurement(os.path.join(directory, file))
        for source in sources:
            if source in sample_data_shape:
                data = meas[source]['data']
                rows = data.shape[0]
                data_mmap[source][offset[source]:offset[source] + rows] = data

            if source in sample_video_shape:
                for i, c_n_v in enumerate(sample_video_shape[source]):
                    video = meas[source]['video'][i]
                    rows = video.shape[0]
                    video_mmap[source][c_n_v][offset[source]:offset[source] +
                                              rows] = video

            time = meas[source]['time']
            time_mmap[source][offset[source]:offset[source] + rows] = time

            offset[source] += rows

    # Flush to ensure all data is written to disk
    for source in sources:
        if source in sample_data_shape:
            data_mmap[source].flush()
        if source in sample_video_shape:
            for c_n_v in sample_video_shape[source]:
                video_mmap[source][c_n_v].flush()

        time_mmap[source].flush()

    measuremnt = {}
    for source in sources:
        measuremnt_ = {}
        if source in sample_data_shape:
            measuremnt_['data'] = data_mmap[source]
            measuremnt_['channel_names'] = meas0[source]['channel_names']

        if source in sample_video_shape:
            measuremnt_['video'] = [
                video_mmap[source][c_n_v]
                for c_n_v in sample_video_shape[source]
            ]
            measuremnt_['channel_names_video'] = meas0[source][
                'channel_names_video']

        measuremnt_['time'] = time_mmap[source]
        measuremnt_['sample_rate'] = meas0[source]['sample_rate']

        measuremnt[source] = measuremnt_

    return measuremnt
