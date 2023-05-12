import os
import numpy as np
import pickle
import datetime
import time
import threading
import copy

import sys
from pyTrigger import pyTrigger

# National Insturments:
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.Task import Task
from nidaqmx._lib import lib_importer
from .daqtask import DAQTask

# Analog Discovery 2:
from . import dwfconstants as dwfc 
from ctypes import *

# Serial communication:
import serial
import struct

from .ni_task import NITask

class DummyLock:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
class CustomPyTrigger(pyTrigger):
    """
    Upgrades pyTrigger class with features needed for acquisition class BaseAcquisition.
    """
    triggered_global = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_acquired_samples               = 0 # samples acquired throughout whole acquisition process
        self.N_acquired_samples_since_trigger = 0 # samples acquired since trigger
        self.N_new_samples                    = 0 # new samples that have not been retrieved yet
        self.N_new_samples_PLOT               = 0 # new samples that have not been retrieved yet - Plotting purposes
        self.N_triggers                       = 0 # amount of time acquisition was triggered (should be 1 at the end of the measurement)
        
        self.first_trigger = True
        
        self.continuous_mode = False # continuous acquisition without definite stop.
        self.N_samples_to_acquire = self.rows # amount of samples to acquire in continuous mode.

    def _add_data_to_buffer(self, data):
        """Upgrades parent _add_data_to_buffer() to track sample variables
           N_acquired_samples, N_new_samples, N_acquired_samples_since_trigger
        """
        if self.continuous_mode:
            if self.triggered and not (self.rows_left > len(data)):
                # if data is larger then rows_left and if continuous mode is enabled,
                # save all data, reset buffer and continue with acquisition
                self.reset_buffer()
                
            if self.N_samples_to_acquire is not None: # if measurement duration is specified
                if self.N_acquired_samples_since_trigger + len(data) >= self.N_samples_to_acquire:
                    data = data[:self.N_samples_to_acquire - self.N_acquired_samples_since_trigger]
                    self.finished = True
            
        rows_left_before = self.rows_left
        super()._add_data_to_buffer(data)
        N = rows_left_before - self.rows_left
        
        self.N_acquired_samples += data.shape[0]
        self.N_new_samples_PLOT += data.shape[0]
        self.N_new_samples      += N
        self.N_acquired_samples_since_trigger += N
        
    def _add_data_chunk(self, data):
        """Upgrades parent _add_data_chunk() to globally trigger all acquisition sources present
           in the measurement process, or that another acquisition source triggers this class.
           Global trigger is implemented via class property variable 'triggered_global'.
        """
        super()._add_data_chunk(data)
        if self.triggered and self.first_trigger:
            CustomPyTrigger.triggered_global = True 
        elif CustomPyTrigger.triggered_global and self.first_trigger:
            self.triggered = True
        else:
            pass

        if self.first_trigger and (self.triggered or CustomPyTrigger.triggered_global):
            self.N_triggers        += 1
            self.first_trigger      = False
        return 
    
    def get_data_new(self):
        """Retrieves any new data from ring buffer after trigger event that has been not yet retrieved.

        Returns:
            np.ndarray: data of shape (rows, channels)
        """
        if self.triggered:
            data = self.ringbuff.get_data()
            if self.N_new_samples > 0:
                data = data[-self.N_new_samples:]
            else:
                data = np.empty(shape=(0, self.channels))
            self.N_new_samples = 0
            
            return data
        else: # NOTE: this should not happen!
            return np.empty(shape=(0, self.ringbuff.columns))
        
    def get_data_new_PLOT(self):
        """Retrieves any new data from ring buffer that has been not yet retrieved. 
           This is used for plotting purposes only.

        Returns:
            np.ndarray: data of shape (rows, channels)
        """
        if self.N_new_samples_PLOT > 0:
            data = self.ringbuff.get_data()[-self.N_new_samples_PLOT:]
            self.N_new_samples_PLOT = 0
            return data
        else:
            return np.empty(shape=(0, self.channels))
    
    def _trigger_index(self, data):
        """Upgrades parent _trigger_index() method. Beside searching for trigger event, it
           adds amount of samples missed by _add_data_to_buffer() in case of use of presamples.
        """
        trigger = super()._trigger_index(data)
        if type(trigger) != np.ndarray:
            self.N_new_samples += self.presamples - trigger # this amount of data will not be added in _add_data_to_buffer()
            self.N_acquired_samples_since_trigger += self.presamples - trigger
        return trigger
    
    def reset_buffer(self):
        self.rows_left = self.rows
        self.finished = False

    
class BaseAcquisition:
    """Parent acquisition class that should be used when creating new child acquisition source class.
    Child class should override methods the following methods:
    - self.read_data()
    - self.terminate_data_source()
    - self.set_data_source()
    - self.clear_buffer()
    - self.get_sample_rate() (optional)
    
    Additionally, the __init__() method should override the following attributes:
    - self.n_channels 
    - self.channel_names 
    - self.sample_rate
    - self.acquisition_name (optional)
    
    NOTE: the __init__() method should call self.set_trigger(1e20, 0, duration=600)
    at the end of __init__ method to set trigger eventhough not used.

    Returns:
        _type_: _description_
    """
    all_acquisitions_ready = False # class property to indicate if all acquisitions are ready to start (not jsut this one)
    
    def __init__(self):
        """EDIT in child class"""
        self.acquisition_name = "DefaultAcquisition"
        self.channel_names = []
        self.is_running = True
        self.is_standalone = True # if this is part of bigger system or used as standalone object
        self.is_ready = False
    
        self.lock_acquisition = threading.Lock() # ensures acquisition class runs properly if used in multiple threads.
        
        self.continuous_mode = False
        self.N_samples_to_acquire = None
        # child class needs to have variables below:
        self.n_channels  = 0
        self.sample_rate = 0
        
    def read_data(self):
        """EDIT in child class
        
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        pass

    def terminate_data_source(self):
        """EDIT in child class
        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        pass

    def set_data_source(self):
        """EDIT in child class
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        pass
    
    def get_sample_rate(self):
        """EDIT in child class
        
        Returns sample rate of acquisition class.
        This function is also useful to compute sample_rate estimation if no sample rate is given
        
        Returns self.sample_rate
        """
        return self.sample_rate
    
    def clear_buffer(self):
        """EDIT in child class
        
        The source buffer should be cleared with this method. Either actually clears the buffer, or
        just reads the data with self.read_data() and does not add/save data anywhere.
        
        Returns None.
        """
        self.read_data()

    # The following methods should work without changing.
    def stop(self):
        """Stops acquisition run.
        """
        self.is_running = False
    
    def acquire(self):
        """Acquires data from acquisition source and also properly saves the data to pyTrigger ringbuffer.
        Additionally it also stops the measurement run and terminates acquisition source properly.
        """
        with self.lock_acquisition: # lock to secure variables
            acquired_data = self.read_data()
            self.Trigger.add_data(acquired_data)
            
        if self.Trigger.finished or not self.is_running:      
        #if not self.is_running:       
            self.stop()
            self.terminate_data_source()

    def get_data(self, N_points=None):
        """Reads and returns data from the pyTrigger buffer.
        :param N_points (int, str, None): number of last N points to read from pyTrigger buffer. 
                            if N_points="new", then only new points will be retrieved.
                            if None all samples are returned.
        Returns:
            tuple: (time, data) - 1D time vector and 2D measured data, both np.ndarray
        """
        
        if N_points is None:
            data = self.Trigger.get_data()[-self.Trigger.N_acquired_samples_since_trigger:]
            
        elif N_points == "new":
            with self.lock_acquisition:
                data = self.Trigger.get_data_new()
        else:
            data = self.Trigger.get_data()[-N_points:]
                
        time = np.arange(data.shape[0])/self.sample_rate     
        return time, data
    
    def get_data_PLOT(self):
        """Reads only new data from pyTrigger ring buffer and returns it.
        NOTE: this method is used only for plotting purposes and should not be used for any other purpose.
              also it does not return time vector, only data.
        Returns:
            array: 2D numpy array of shape (N_new_samples, n_channels)
        """
        with self.lock_acquisition:
            return self.Trigger.get_data_new_PLOT()
    
    def get_measurement_dict(self, N_points=None):
        """Reads data from pyTrigger ring buffer using self.get_data() method and returns a dictionary
           {'data': data, 'time': time, 'channel_names': self.channel_names, 'sample_rate' : sample_rate}

        Args:
            N_points (None, int, str): Number fo points to get from pyTrigger ringbuffer. If type(N_points)==int then N_points
                                       last samples are returned. If N_points=='new', only new points after trigger event are returned.
                                       If None, all samples are returned. Defaults to None.

        Returns:
            dict: {'data': data, 'time': time, 'channel_names': self.channel_names, 'sample_rate' : sample_rate}
        """
        if N_points is None:
            time, data = self.get_data(N_points=N_points)
        else:
            time, data = self.get_data(N_points=N_points)
        
        self.measurement_dict = {
            'data': data,
            'time': time,
            'channel_names': self.channel_names,
            'sample_rate' : None, 
        }
        
        if hasattr(self, 'sample_rate'):
            self.measurement_dict['sample_rate'] = self.sample_rate
            
        return self.measurement_dict
        
    def run_acquisition(self, run_time=None):
        """
        Runs acquisition.
        :params: run_time - (float) number of seconds for which the acquisition will run. 
            If None acquisition runs indefinitely until self.is_running variable is set
            False externally (i.e. in a different process)
        """
        BaseAcquisition.all_acquisitions_ready = False 
        self.is_ready = False
        self.is_running = True
        
        if run_time is None:
            self._set_trigger_instance()
        else:
            self.update_trigger_parameters(duration=run_time, duration_unit='seconds')
        self.set_data_source()
        
        # if acquisition is used in some other classes, wait until all acquisition sources are ready:
        if not self.is_standalone:
            self.is_ready = True    # this source is ready (other may not be)
            while not BaseAcquisition.all_acquisitions_ready: # until every source is ready
                time.sleep(0.01)
                self.clear_buffer()                           # reads data, does not store in anywhere
                if not self.is_running:
                    break
                
            time.sleep(0.01)
            self.clear_buffer() # ensure buffer is cleared at least once. 
        
        self.actual_run_time = 0 # actual time run to obtain all samples
        time_start_acq = time.time()
        
        # main acquisition loop:
        if run_time == None:
            while self.is_running:
                time.sleep(0.01)
                self.acquire()
        else:
            time_start = time.time()
            while self.is_running:  
                if time_start + run_time < time.time():
                    self.is_running = False

                time.sleep(0.01)
                self.acquire()
                
        # save actual measurement time (NOTE: currently not used anywhere, might be useful in the future)
        time_end_acq  = time.time()
        self.actual_run_time = time_end_acq-time_start_acq
       
    def set_continuous_mode(self, boolean=True, measurement_duration=None):
        """Sets continuous mode of the acquisition. If True, acquisition will run indefinitely until
           externally stopped. If False, acquisition will run for a specified time.

        Args:
            boolean (bool, optional): Defaults to True.
            measurement_duration (float, optional): If not None, sets the duration of the measurement in seconds.
            NOTE: Based on measurement duration, the number of total samples to be acquired is calculated. In this case the 
            ring buffer size can be different to the number of samples to be acquired. If None, measurement duration is 
            set to the size of the ring buffer.
        """
        if boolean:
            self.continuous_mode = True
        else:
            self.continuous_mode = False
            
        if measurement_duration is not None:
            self.N_samples_to_acquire = int(measurement_duration*self.sample_rate)
                   
    def _set_trigger_instance(self):
        """Creates PyTrigger instance.
        """
        self.Trigger = CustomPyTrigger( #pyTrigger
            rows=self.trigger_settings['duration_samples'], 
            channels=self.n_channels,
            trigger_type=self.trigger_settings['type'],
            trigger_channel=self.trigger_settings['channel'], 
            trigger_level=self.trigger_settings['level'],
            presamples=self.trigger_settings['presamples'])
        
        self.Trigger.continuous_mode = self.continuous_mode
        if self.continuous_mode:
            self.Trigger.N_samples_to_acquire = self.N_samples_to_acquire           
        
    def set_trigger(self, level, channel, duration=1, duration_unit='seconds', presamples=0, type='abs'):
        """Set parameters for triggering the measurement.
        
        :param level: trigger level
        :param channel: trigger channel
        :param duration: durtion of the acquisition after trigger (in seconds or samples)
        :param duration_unit: 'seconds' or 'samples'
        :param presamples: number of presampels to save
        :param type: trigger type: up, down or abs"""

        if duration_unit == 'seconds':
            duration_samples = int(self.sample_rate*duration)
            duration_seconds = duration
        elif duration_unit == 'samples':
            duration_samples = int(duration)
            duration_seconds = duration/self.sample_rate

        self.trigger_settings = {
            'level': level,
            'channel': channel,
            'duration': duration,
            'duration_unit': duration_unit,
            'presamples': presamples,
            'type': type,
            'duration_samples': duration_samples,
            'duration_seconds': duration_seconds,
        }
        
        self._set_trigger_instance()
        
    def update_trigger_parameters(self, **kwargs):
        """
        Updates trigger settings. See 'set_trigger' method for possible parameters.
        """  
        for setting, value in kwargs.items():
            self.trigger_settings[setting] = value
            
        if self.trigger_settings['duration_unit'] == 'seconds':
            self.trigger_settings['duration_samples'] = int(self.sample_rate*self.trigger_settings['duration'])
            self.trigger_settings['duration_seconds'] = self.trigger_settings['duration']
         
        elif self.trigger_settings['duration_unit'] == 'samples':
            self.trigger_settings['duration_seconds'] = self.trigger_settings['duration']/self.sample_rate
            self.trigger_settings['duration_samples'] = self.trigger_settings['duration']
        
        self._set_trigger_instance()
        
    def activate_trigger(self, all_sources=True):
        """Sets trigger off. Useful if the acquisition class is trigered by another process.
            This trigger can also trigger other acquisition sources by setting property class
        """
        if all_sources:
            CustomPyTrigger.triggered_global = True
        else:
            self.Trigger.triggered = True

    def reset_trigger(self):
        """Resets trigger.
        """
        CustomPyTrigger.triggered_global = False
        self.Trigger.triggered = False
        
    def is_triggered(self):
        """Checks if acquisition class has been triggered during measurement.

        Returns:
            bool: True/False if triggered
        """
        return self.Trigger.triggered
        
    def _all_acquisitions_ready(self):
        """Sets ALL acquisition sources (not only this one) to ready state. Should not be generally used.
        """
        BaseAcquisition.all_acquisitions_ready = True
    
    def save(self, name, root='', timestamp=True, comment=None):
        """Save acquired data.
        
        :param name: filename
        :param root: directory to save to
        :param timestamp: include timestamp before 'filename'
        :param comment: commentary on the saved file
        """
        self.measurement_dict = self.get_measurement_dict()
        
        if comment is not None:
            self.measurement_dict['comment'] = comment
        
        if not os.path.exists(root):
            os.mkdir(root)

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        filename = f'{stamp}{name}.pkl'
        path = os.path.join(root, filename)
        pickle.dump(self.measurement_dict, open(path, 'wb'), protocol=-1)


class WaveFormsAcquisition(BaseAcquisition):
    def __init__(self, channels=[0, 1], sample_rate=10000, 
                 channel_names=None, acquisition_name=None, device_number=None):
        super().__init__()

        self.acquisition_name = 'AD2' if acquisition_name is None else acquisition_name
        self.channel_names = channel_names if channel_names is not None else [f'CH{i}' for i in channels]
        self.channel_idx = channels
        
        self.n_channels  = len(channels)
        self.sample_rate = sample_rate
        self.device_number = device_number if device_number is not None else -1
        
        if sys.platform.startswith("win"):
            self.dwf = cdll.dwf
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

        self.hdwf = c_int(0) # device handle
        
        # tracking of lost data and corrupted data:
        self.cLost = c_int()
        self.cCorrupted = c_int()
        self.fLost = 0
        self.fCorrupted = 0
        
        #self.set_data_source()
        self.configure_channels() # configure channel range
        
        self.set_trigger(1e20, 0, duration=600)
        
        
    def configure_channels(self, input_range=None):
        """Specify min and max value range for each channel.
        Args:
        input_range (dict): dictionary with channel index as key and tuple of min and max values as value. channel indices
                            have to be the same as self.channel_idx (or channels input parameters in init)
                            For example: {0:(-10, 10), 1:(-5, 5)} 
                            -> channel 0 has range -10 to 10 V and channel 1 has range -5 to 5 V.
                            
        """
        if input_range is None:
            if not hasattr(self, 'input_range'):
                input_range = {idx:(-10, 10) for idx in self.channel_idx}
                self.input_range = input_range
                    
        # based on which channels are used:
        for idx in self.channel_idx:
            val_min, val_max = self.input_range[idx]
            ch_range = val_max - val_min
            ch_offset = (val_max + val_min)/2
            
            # enable channel:
            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(idx), c_bool(True))
            # set range:
            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(idx), c_double(ch_range))
            # set offset:
            self.dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, c_int(idx), c_double(ch_offset))
        
        
    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        sts = c_byte() # acquisition status
        cAvailable = c_int() # number of samples available
        
        self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))
        if (sts == dwfc.DwfStateConfig or sts == dwfc.DwfStatePrefill or sts == dwfc.DwfStateArmed) :
            # Acquisition not yet started.
            return np.empty((0, self.n_channels))

        self.dwf.FDwfAnalogInStatusRecord(self.hdwf, byref(cAvailable), byref(self.cLost), byref(self.cCorrupted))
        if self.cLost.value :
            self.fLost = 1
        if self.cCorrupted.value :
            self.fCorrupted = 1

        if cAvailable.value==0: # no data available
            return np.empty((0, self.n_channels))
        
        arr = []
        for i in self.channel_idx:
            rgdSamples = (c_double*cAvailable.value)()
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(i), byref(rgdSamples), cAvailable) # get channel 1 data
            values = np.fromiter(rgdSamples, dtype =float)
            arr.append(values)
        arr = np.array(arr).T
        return arr

   
    def set_data_source(self):
        """
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        
        if self.hdwf.value == dwfc.hdwfNone.value: # if device is not open
            self.dwf.FDwfDeviceOpen(self.device_number, byref(self.hdwf))

        self.dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, dwfc.acqmodeRecord)
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.sample_rate))
        self.dwf.FDwfAnalogInRecordLengthSet(self.hdwf, c_double(-1)) # -1 infinite record length
        
        self.configure_channels()
        #wait at least 2 seconds for the offset to stabilize
        time.sleep(0.3)

        # check if the device is running:
        device_state = c_int()
        self.dwf.FDwfAnalogInStatus(self.hdwf, c_bool(True), byref(device_state)) 
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))

        # Temporary solution for setting generation signal:
        #print("Generating")
        #self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf,   c_int(0), dwfc.AnalogOutNodeCarrier, c_bool(True))
        #self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_int(0), dwfc.AnalogOutNodeCarrier, dwfc.funcSine)
        #self.dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf,c_int(0), dwfc.AnalogOutNodeCarrier, c_double(100))
        #self.dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf,c_int(0), dwfc.AnalogOutNodeCarrier, c_double(2))
        #self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, c_int(0), dwfc.AnalogOutNodeCarrier, c_double(0))
        #elf.dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_bool(True))
     
    
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        #self.dwf.FDwfAnalogOutReset(self.hdwf, c_int(0))
        self.dwf.FDwfDeviceCloseAll()
        self.hdwf = c_int(0)
        pass

    
    def get_sample_rate(self):
        """
        Returns sample rate of acquisition class.
        This function is also useful to compute sample_rate estimation if no sample rate is given
        
        Returns self.sample_rate
        """
        return self.sample_rate
    
    def clear_buffer(self):
        """
        The source buffer should be cleared with this method. Either actually clears the buffer, or
        just reads the data with self.read_data() and does not add/save data anywhere.
        
        Returns None.
        """
        self.read_data()

class SerialAcquisition(BaseAcquisition):
    """General Purpose Class for Serial Communication"""
    def __init__(self, port, baudrate, byte_sequence, timeout=1, start_bytes=b"\xfa\xfb", end_bytes=b"\n", 
                       write_start_bytes=None, write_end_bytes=None, pretest_time=None, sample_rate=None,
                       channel_names = None, acquisition_name=None ):
        """
        Initializes serial communication.

        :param: port - (str) serial port (i.e. "COM1")
        :param: baudrate - (int) baudrate for serial communication
        :param: byte_sequence - (tuple) data sequence in each recived line via serial communication
                                example: (("int16", 2), ("int32", 2), ("uint16", 3))
                                explanations: line consists of 2 16bit signed intigers, followed by 
                                2 signed 32bit intigers, followed by 3 unsigned 16bit intigers.

                                supported types: int8, uint8, int16, uint16, int32, uint32
        :param: start_bytes - (bstr) received bytes via serial communication indicating the start of each line
        :param: end_bytes   - (bstr) recieved bytes via serial communication indicating the end of each line
        :param: write_start_bytes - bytes to be written at the beggining of acquisition
                                    if (list/tuple/byte/bytearray) sequence of b"" strings with bytes to write to initiate/set data transfer or other settings
                                    Writes each encoded bstring with 10 ms delay.
                                    if list/tuple, then elements have to be of type byte/bytearray
        :param: write_end_bytes   - bytes to be written at the beggining of acquisition
                                    if (list/tuple/byte/bytearray) sequence of b"" strings with bytes to write to initiate/set data transfer or other settings
                                    Writes each encoded bstring with 10 ms delay.
                                    if list/tuple, then elements have to be of type byte/bytearray    
        :param: pretest_time - (float) time for which sample rate test is run for when class is created. If None, 10 second spretest is performed
        :param: sample_rate  - (float) Sample rate at which data is acquired. If None, then sample_rate pretest will be performed for 'pretest_time' seconds.
        :param: channel_names - (list) list of strings of channel names. Defaults to None.  
        """
        super().__init__()
        if acquisition_name is None:
            self.acquisition_name = "SerialAcquisition"
        else:
            self.acquisition_name = acquisition_name

        self.port = port
        self.baudrate = baudrate
        self.byte_sequence = byte_sequence
        self.start_bytes_write = write_start_bytes
        self.end_bytes_write   = write_end_bytes
        self.start_bytes = start_bytes
        self.end_bytes = end_bytes
        self.timeout   = timeout

        self.unpack_string = b""
        self.expected_number_of_bytes = 0
        self.n_channels = 0
        self.channel_names = channel_names
    
        self.set_unpack_data_settings() # sets unpack_string, expected_number_of_bytes, n_channels
        self.set_channel_names()        # sets channel names if none were given to the class
        
        self.set_data_source()          # initializes serial connection
    
        self.buffer = b""                # buffer to which recieved data is added

        # Estimate sample_rate:
        self.pretest_time = pretest_time if pretest_time is not None else 10.
        self.sample_rate = sample_rate if sample_rate is not None else self.get_sample_rate()
        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def set_data_source(self):
        # open terminal:
        if not hasattr(self, 'ser'):
            try:
                self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, 
                                         timeout=self.timeout)
            except serial.SerialException:
                print("Serial port is in use.")
        elif not self.ser.is_open:
            self.ser.open()
            time.sleep(1.0)
        else:
            pass 
        
        # Send commands over serial:
        self.write_to_serial(self.start_bytes_write)
        time.sleep(0.5)

        self.ser.reset_input_buffer() # clears previous data
        self.buffer = b"" # reset class buffer
        
    def terminate_data_source(self):
        self.buffer = b""
        time.sleep(0.01)
        self.write_to_serial(self.end_bytes_write)
        time.sleep(0.1)
        self.ser.close()

    def read_data(self):
        # 1) read all data from serial
        self.buffer += self.ser.read_all()

        # 2) split data into lines
        parsed_lines = self.buffer.split(self.end_bytes + self.start_bytes)
        if len(parsed_lines) == 1 or len(parsed_lines) == 0: # not enough data
            return np.array([]).reshape(-1, self.n_channels)
        
        # 3) decode full lines, convert data to numpy array
        data = []
        for line in parsed_lines[:-1]: # last element probably does not contain all data
            if len(line) == self.expected_number_of_bytes - len(self.end_bytes+self.start_bytes):
                line_decoded = struct.unpack(self.unpack_string, line)
                data.append(line_decoded)
            else:
                #print(f"Expected nr. of bytes {self.expected_number_of_bytes}, line contains {len(line)}")
                pass
        data = np.array(data)
        if len(data) == 0:
            data = data.reshape(-1, self.n_channels)

        # 4) reset buffer with remaninig bytes:
        self.buffer = self.end_bytes + self.start_bytes + parsed_lines[-1]

        return data
    
    def clear_buffer(self):
        """Clears serial buffer.
        """
        self.ser.read_all()

    def set_unpack_data_settings(self):
        """
        Converts byte_sequence to string passed to struct unpack method.
        """
        self.convert_dict = {
            "uint8":  ("B", 1), # (struct format, number of bytes)
            "int8":   ("b", 1),
            "uint16": ("H", 2),
            "int16":  ("h", 2),
            "uint32": ("L", 4),
            "int32":  ("l", 4),
            "float":  ("f", 4)
        }

        self.unpack_string = "<" # order of several bytes for 1 variable (see struct library)
        self.n_channels = 0
        self.expected_number_of_bytes = len(self.start_bytes) + len(self.end_bytes)
        for seq in self.byte_sequence:
            dtype, n = seq
            for i in range(n):
                self.unpack_string += self.convert_dict[dtype][0]
                self.expected_number_of_bytes += self.convert_dict[dtype][1]
                self.n_channels += 1

        return self.unpack_string

    def set_channel_names(self):
        """
        Sets default channel names if none were passed to the class.
        """
        if self.channel_names is None:
            self.channel_names = [f"channel {i+1}" for i in range(self.n_channels)]
        else:
            if len(self.channel_names) != self.n_channels:
                self.channel_names = [f"channel {i+1}" for i in range(self.n_channels)]
            else:
                self.channel_names = self.channel_names

    def write_to_serial(self, write_bytes):
        """
        Writes data to serial port.

        :param: write_start_bytes - bytes to be written at the beggining of acquisition
                                if (list/tuple/byte/bytearray) sequence of b"" strings with bytes to write to initiate/set data transfer or other settings
                                Writes each encoded bstring with 10 ms delay.
                                if list/tuple, then elements have to be of type byte/bytearray
        """
        if write_bytes is None:
            pass
        else:
            if isinstance(write_bytes, list):
                if all(isinstance(b, (bytes, bytearray)) for b in write_bytes):
                    for byte in write_bytes:
                        self.ser.write(byte)
                        time.sleep(0.01)
                else:
                    raise TypeError("write_bytes have to be bytes or bytearray type.")

            elif isinstance(write_bytes, (bytes, bytearray)):
                self.ser.write(write_bytes)
                time.sleep(0.01)
            else:
                raise TypeError("write_bytes have to be bytes or bytearray type.")
            
    def get_sample_rate(self):
        self.set_data_source()
        time.sleep(0.1)
                
        # Run pretest:
        self.is_running = True
        self.buffer = b""
        
        # pretest:
        print(f"Running pretest to estimate sample rate for {self.pretest_time} seconds...")
        time_start = time.time()
        n_cycles = 0
        while True:
            self.buffer += self.ser.read_all()
            n_cycles    += 1
            if time.time()-time_start >= self.pretest_time:
                break
        time_end = time.time()
        self.buffer += self.ser.read_all()
        
        self.buffer2 = self.buffer
        # parse data:
        parsed_lines = self.buffer.split(self.end_bytes + self.start_bytes)
        if len(parsed_lines) == 1 or len(parsed_lines) == 0: # not enough data
            print(ValueError(f"No data has been transmitted. Sample rate {self.sample_rate} Hz will be assumed."))
            self.sample_rate = 100
        else:
            data = []
            for line in parsed_lines[:-1]: # last element probably does not contain all data
                if len(line) == self.expected_number_of_bytes - len(self.end_bytes+self.start_bytes):
                    line_decoded = struct.unpack(self.unpack_string, line)
                    data.append(line_decoded)
                else:
                    pass
            self.Trigger.N_acquired_samples = len(data)
            
            # calculate sample_rate:
            self.sample_rate = int( self.Trigger.N_acquired_samples / (time_end - time_start ) )
            
            # this is overcomplicating things:   :)
            t_cycle = (time_end - time_start )/n_cycles
            self.sample_rate = int( (self.Trigger.N_acquired_samples + t_cycle*self.sample_rate) / (time_end - time_start ) )
        
        if self.sample_rate == 0:
            print("Something went wrong. Please check 'byte_sequence' input parameter if recieved byte sequence is correct.")
            
        # end acquisition:
        self.stop()
        self.terminate_data_source()
        print("Completed.")
        return self.sample_rate
    
class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class."""

    def __init__(self, task_name, acquisition_name=None):
        """Initialize the task.

        :param task_name: Name of the task from NI Max
        """
        super().__init__()

        try:
            DAQmxClearTask(taskHandle_acquisition)
        except:
            pass

        try:
            lib_importer.windll.DAQmxClearTask(taskHandle_acquisition)
        except:
            pass
        
        self.task_terminated = True

        self.task_base = task_name
        if isinstance(task_name, str):
            self.NITask_used = False
        elif isinstance(task_name, NITask):
            self.NITask_used = True
        else:
            raise TypeError("task_name has to be a string or NITask object.")

        self.set_data_source()
        self.acquisition_name = self.task_name if acquisition_name is None else acquisition_name

        self.sample_rate = self.Task.sample_rate
        self.channel_names = self.Task.channel_list
        self.n_channels = self.Task.number_of_ch

        if not self.NITask_used:
            glob_vars = globals()
            glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def clear_task(self):
        """Clear a task."""
        self.Task.clear_task(wait_until_done=False)
        time.sleep(0.1)
        del self.Task

    def terminate_data_source(self):
        self.task_terminated = True
        self.clear_task()
        
    def read_data(self):
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def clear_buffer(self):
        self.Task.acquire_base()
    
    def set_data_source(self):
        if self.task_terminated:
            if self.NITask_used:
                channels_base = copy.deepcopy(self.task_base.channels)
                self.Task = NITask(self.task_base.task_name, self.task_base.sample_rate, self.task_base.settings_file)
                self.task_name = self.task_base.task_name

                for channel_name, channel in channels_base.items():
                    self.Task.add_channel(
                        channel_name, 
                        channel['device_ind'],
                        channel['channel_ind'],
                        channel['sensitivity'],
                        channel['sensitivity_units'],
                        channel['units'],
                        channel['serial_nr'],
                        channel['scale'],
                        channel['min_val'],
                        channel['max_val'])
            else:
                self.Task = DAQTask(self.task_base)
            
            self.task_terminated = False
        
        if self.NITask_used:
            if not hasattr(self.Task, 'task'):
                self.Task.initiate()

    def run_acquisition(self, run_time=None):        

        if self.NITask_used:
            BaseAcquisition.all_acquisitions_ready = False 
            self.is_ready = False
            self.is_running = True
            
            if run_time is None:
                self._set_trigger_instance()
            else:
                self.update_trigger_parameters(duration=run_time, duration_unit='seconds')
            
            self.set_data_source()
            glob_vars = globals()
            glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        super().run_acquisition(run_time)


