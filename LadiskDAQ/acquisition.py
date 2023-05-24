import os
import numpy as np
import pickle
import datetime
import time
import threading
import copy

import sys
from pyTrigger import pyTrigger, RingBuffer2D

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

try:
    import PySpin # This will be moved somewhere else in the future
except:
    print("PySpin library not found.")
    
try:    
    from pypylon import genicam, pylon
except:
    print("pypylon library not found. Please install using pip install pypylon")

    
class CustomPyTrigger(pyTrigger):
    """
    Upgrades pyTrigger class with features needed for acquisition class BaseAcquisition.
    
    :param rows: # of rows
    :param channels: # of channels
    :param trigger_channel: the channel used for triggering
    :param trigger_level: the level to cross, to start trigger
    :param trigger_type: 'up' is default, possible also 'down'/'abs'
    :param presamples: # of presamples
    """
    triggered_global = False
    #def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
    def __init__(self, rows=5120, channels=4, trigger_channel=0,
                 trigger_level=1., trigger_type='up', presamples=1000,
                 dtype=np.float64):
        
        self.rows = rows
        self.channels = channels
        self.trigger_channel = trigger_channel
        self.trigger_level = trigger_level
        self.trigger_type = trigger_type.lower()
        self.presamples = presamples
        self.ringbuff =  RingBuffer2D(rows=self.rows, columns=self.channels, dtype=dtype)
        self.triggered = False
        self.rows_left = self.rows
        self.finished = False
        self.first_data = True
        
        
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
        self.buffer_dtype = np.float64 # default dtype of data in ring buffer
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
        else:
            # acquisition is being run as a standalone process, so no need to wait for other sources
            pass
        
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
            presamples=self.trigger_settings['presamples'],
            dtype=self.buffer_dtype)
        
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

class IRFormatType:
    LINEAR_10MK = 1
    LINEAR_100MK = 2
    RADIOMETRIC = 3
    
class FLIRThermalCamera(BaseAcquisition):    
    """
    Acquisition class for FLIR thermal camera (A50)
    
    This class is adapted from examples for thermal A50 camera provided by FLIR, found on their website (login required):
    https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/
    
    Installation steps:
    1) Install Spinnaker SDK (i.e. SpinnakerSDK_FULL_3.1.0.79_x64.exe, found on provided link)
    2) Install PySpin (python wrapper for Spinnaker SDK)
    """
    def __init__(self, acquisition_name=None, buffer_dtype=np.float16):
        super().__init__()
        self.buffer_dtype = buffer_dtype
        self.acquisition_name = 'FLIR' if acquisition_name is None else acquisition_name
        self.image_shape = None
        
        self.set_IRtype('LINEAR_10MK')
        self.camera_acq_started = False
        self.set_data_source()
        # self.terminate_data_source()
        #print(self.image_shape)
        
        # there will always be only 1 channel and it will always display temperature
        self.n_channels  = self.image_shape[0]*self.image_shape[1]
        self.channel_names = ['Temperature']
        self.sample_rate = 30 # TODO: this can probably be set in thermal camera and read from it
                              # default camera fps is 30.
        
        # channel in set trigger is actually pixel in flatten array:
        self.set_trigger(1e20, 0, duration=1.0)
        
        
        # TODO:
        # - set sample rate (either subsample and only acquire every n-th frame or set camera fps)
        # - adjust picture resolution
        
    def set_IRtype(self, IRtype):
        '''This function sest the IR type to be used by the camera.

        Sets the IR type to either:
            - LINEAR_10MK: 10mK temperature resolution
            - LINEAR_100MK: 100mK temperature resolution
            - RADIOMETRIC: capture radiometric data and manually convert to temperature

        Parameters
        ----------
        IRtype : str
            IR type to be used by the camera (either LINEAR_10MK, LINEAR_100MK or RADIOMETRIC)
        '''
        avaliable_types = [
            i for i in IRFormatType.__dict__.keys() if i[:1] != '_'
        ]
        if IRtype not in avaliable_types:
            raise ValueError(
                f'IRtype must be one of the following: {avaliable_types}')
        self.CHOSEN_IR_TYPE = getattr(IRFormatType, IRtype)
        
    def set_data_source(self):
        """
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        if hasattr(self, 'cam'):
            return # we already have a camera set up
        
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()
        # Get current library version
        version = self.system.GetLibraryVersion()
        #print('Library version: %d.%d.%d.%d' %
        #      (version.major, version.minor, version.type, version.build))

        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        #print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            raise ValueError('No cameras detected!')

        elif num_cameras > 1:
            # Clear camera list before releasing system
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            raise ValueError('More than one camera detected!')
        elif num_cameras < 0:
            raise ValueError('Something went wrong with camera detection!')
        
        # we have exactly one camera:
        self.cam = self.cam_list.GetByIndex(0)
        # Initialize camera
        self.cam.Init()
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        # Retrieve GenICam nodemap
        self.nodemap = self.cam.GetNodeMap()
        
        sNodemap = self.cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
        node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
        
        pixel_format_mono16 = node_pixel_format_mono16.GetValue()
        node_pixel_format.SetIntValue(pixel_format_mono16)
        #print("pixelformat:", pixel_format_mono16)
        
        if self.CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK: # LINEAR_10MK
            # This section is to be activated only to set the streaming mode to TemperatureLinear10mK
            node_IRFormat = PySpin.CEnumerationPtr(self.nodemap.GetNode('IRFormat'))
            node_temp_linear_high = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear10mK'))
            node_temp_high = node_temp_linear_high.GetValue()
            node_IRFormat.SetIntValue(node_temp_high)
        elif self.CHOSEN_IR_TYPE == IRFormatType.LINEAR_100MK: # LINEAR_100MK
            # This section is to be activated only to set the streaming mode to TemperatureLinear100mK
            node_IRFormat = PySpin.CEnumerationPtr(self.nodemap.GetNode('IRFormat'))
            node_temp_linear_low = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear100mK'))
            node_temp_low = node_temp_linear_low.GetValue()
            node_IRFormat.SetIntValue(node_temp_low)
        elif self.CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC: # RADIOMETRIC
            # This section is to be activated only to set the streaming mode to Radiometric
            node_IRFormat = PySpin.CEnumerationPtr(self.nodemap.GetNode('IRFormat'))
            node_temp_radiometric = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('Radiometric'))
            node_radiometric = node_temp_radiometric.GetValue()
            node_IRFormat.SetIntValue(node_radiometric)
            
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            self.terminate_data_source()
            raise ValueError('Unable to set stream buffer handling mode.')
        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            self.terminate_data_source()
            raise ValueError('Unable to set stream buffer handling mode.')
        
        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()
        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
        
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                self.terminate_data_source()
                raise ValueError('Unable to set acquisition mode to continuous.')   

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
                self.terminate_data_source()
                raise ValueError('Unable to set acquisition mode to continuous (entry retrieval).')

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            self.cam.BeginAcquisition()
            self.camera_acq_started = True

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            # Retrieve Calibration details
            self.calib_dict = {}
            CalibrationQueryR_node = PySpin.CFloatPtr(self.nodemap.GetNode('R'))
            self.calib_dict['R'] = CalibrationQueryR_node.GetValue()

            CalibrationQueryB_node = PySpin.CFloatPtr(self.nodemap.GetNode('B'))
            self.calib_dict['B'] = CalibrationQueryB_node.GetValue()

            CalibrationQueryF_node = PySpin.CFloatPtr(self.nodemap.GetNode('F'))
            self.calib_dict['F'] = CalibrationQueryF_node.GetValue()

            CalibrationQueryX_node = PySpin.CFloatPtr(self.nodemap.GetNode('X'))
            self.calib_dict['X'] = CalibrationQueryX_node.GetValue()

            CalibrationQueryA1_node = PySpin.CFloatPtr(self.nodemap.GetNode('alpha1'))
            self.calib_dict['A1'] = CalibrationQueryA1_node.GetValue()

            CalibrationQueryA2_node = PySpin.CFloatPtr(self.nodemap.GetNode('alpha2'))
            self.calib_dict['A2'] = CalibrationQueryA2_node.GetValue()

            CalibrationQueryB1_node = PySpin.CFloatPtr(self.nodemap.GetNode('beta1'))
            self.calib_dict['B1'] = CalibrationQueryB1_node.GetValue()

            CalibrationQueryB2_node = PySpin.CFloatPtr(self.nodemap.GetNode('beta2'))
            self.calib_dict['B2'] = CalibrationQueryB2_node.GetValue()

            CalibrationQueryJ1_node = PySpin.CFloatPtr(self.nodemap.GetNode('J1'))  # Gain
            self.calib_dict['J1'] = CalibrationQueryJ1_node.GetValue()

            CalibrationQueryJ0_node = PySpin.CIntegerPtr(self.nodemap.GetNode('J0'))  # Offset
            self.calib_dict['J0'] = CalibrationQueryJ0_node.GetValue()
            
            
            if self.CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
                # Object Parameters. For this demo, they are imposed!
                # This section is important when the streaming is set to radiometric and not TempLinear
                # Image of temperature is calculated computer-side and not camera-side
                # Parameters can be set to the whole image, or for a particular ROI (not done here)
                Emiss = 0.97
                TRefl = 293.15
                TAtm = 293.15
                TAtmC = TAtm - 273.15
                Humidity = 0.55

                Dist = 2
                ExtOpticsTransmission = 1
                ExtOpticsTemp = TAtm
                
                R, B, F, X, A1, A2, B1, B2, J1, J0 = (self.calib_dict['R'], self.calib_dict['B'], self.calib_dict['F'], self.calib_dict['X'], 
                                                      self.calib_dict['A1'], self.calib_dict['A2'], self.calib_dict['B1'], self.calib_dict['B2'], 
                                                      self.calib_dict['J1'], self.calib_dict['J0'])
                
                H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC -
                                        0.00027816 * TAtmC * TAtmC +
                                        0.00000068455 * TAtmC * TAtmC * TAtmC)

                Tau = X * np.exp(-np.sqrt(Dist) *
                                 (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(
                                     -np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))

                # Pseudo radiance of the reflected environment
                r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))

                # Pseudo radiance of the atmosphere
                r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))

                # Pseudo radiance of the external optics
                r3 = ((1 - ExtOpticsTransmission) /
                      (Emiss * Tau * ExtOpticsTransmission)) * (
                          R / (np.exp(B / ExtOpticsTemp) - F))

                K2 = r1 + r2 + r3
                self.calib_dict['K2'] = K2
                self.calib_dict['Emiss'] = Emiss
                self.calib_dict['Tau'] = Tau
                
            self.node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
            self.node_height = PySpin.CIntegerPtr(self.nodemap.GetNode('Height'))
            self.offsetX = PySpin.CIntegerPtr(self.nodemap.GetNode('OffsetX'))
            self.offsetY = PySpin.CIntegerPtr(self.nodemap.GetNode('OffsetY'))
            self.image_shape = (self.node_height.GetMax() - self.offsetY.GetMax(), self.node_width.GetValue()-self.offsetX.GetMax())
        
        except PySpin.SpinnakerException as ex:
            raise Exception('Error: %s' % ex)

    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        
     
        image_result = self.cam.GetNextImage()
        #  Ensure image completion
        if image_result.IsIncomplete():
            return  np.empty((0, self.n_channels))

        # Getting the image data as a np array
        image_data = image_result.GetNDArray()
        if self.CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK:
            # Transforming the data array into a temperature array, if streaming mode is set to TemperatueLinear10mK
            image_Temp_Celsius_high = (image_data *
                                        0.01) - 273.15
            image_temp = image_Temp_Celsius_high
            
        elif self.CHOSEN_IR_TYPE == IRFormatType.LINEAR_100MK:
            # Transforming the data array into a temperature array, if streaming mode is set to TemperatureLinear100mK
            image_Temp_Celsius_low = (image_data * 0.1) - 273.15
            image_temp = image_Temp_Celsius_low

        elif self.CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
            # Transforming the data array into a pseudo radiance array, if streaming mode is set to Radiometric.
            # and then calculating the temperature array (degrees Celsius) with the full thermography formula
            J0, J1, B, R, Emiss, Tau, K2, F = (self.calib_dict["J0"], self.calib_dict["J1"], self.calib_dict["B"], 
                                            self.calib_dict["R"], self.calib_dict["Emiss"], self.calib_dict["Tau"], 
                                            self.calib_dict["K2"], self.calib_dict["F"])
            image_Radiance = (image_data - J0) / J1
            image_temp = (B / np.log(R / ( (image_Radiance / Emiss / Tau) - K2) + F) ) - 273.15
        else:
            raise Exception('Unknown IRFormatType')
        
        image_temp = image_temp.reshape(-1, self.n_channels)
        
        if image_temp.shape[0] > 0:
            image_result.Release()
            return image_temp
        else:
            return np.empty((0, self.n_channels))
    
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        if self.camera_acq_started:
            self.cam.EndAcquisition()
            self.camera_acq_started = False
            
        self.cam.DeInit()
        del self.cam
        # Clear camera list before releasing system
        self.cam_list.Clear()
        # Release system instance
        self.system.ReleaseInstance()
   
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
        
    def get_data(self, N_points=None):
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        """
        time, data = super().get_data(N_points=N_points)
        data = data.reshape(data.shape[0], self.image_shape[0], self.image_shape[1])
        return time, data
    
    def get_data_PLOT(self):
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        """
        data = super().get_data_PLOT()
        data = data.reshape(data.shape[0], self.image_shape[0], self.image_shape[1])
        return data

class BaslerCamera(BaseAcquisition):
    """
    Acquisition class for Basler camera using pypylon library.
    
    Link to required programs:
    https://www.baslerweb.com/en/downloads/software-downloads/#type=pylonsoftware;language=all;version=7.3.0
    
    Installation steps:
    1) Download and install pylon 7.3.0 Camera Software Suite Windows software and choose developer option during installation
    2) Install python library with pip install pypylon
    
    """
    def __init__(self, acquisition_name=None, sample_rate=60, offset=(0, 0), size=(4112, 3008),
                 subsample=1, pixel_format="Mono12", exposure_time_ms=4.0):
        super().__init__()

        self.acquisition_name = 'BaslerCamera' if acquisition_name is None else acquisition_name
        self.image_shape = None
        
        self.sample_rate = sample_rate # camera fps
        self.subsample = subsample # subsample factor to reduce resolution
        self.size = size # camera size
        self.offset = offset # camera offsets
        self.pixel_format = pixel_format
        self.exposure_time = exposure_time_ms # in ms
        
        self.buffer_dtype = np.uint16# TODO: adjust this according to pixel_format
        self.camera_acq_started = False
        self.set_data_source(start_grabbing=False) # to read self.image_shape
        
        # there will always be only 1 channel and it will always display temperature
        self.n_channels  = self.image_shape[0]*self.image_shape[1]
        self.channel_names = ['Camera']
        
        
        # channel in set trigger is actually pixel in flatten array:
        self.set_trigger(1e20, 0, duration=1.0)
       
    def set_data_source(self, start_grabbing=True):
        """
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        self.current_image_ID = 0 # reset image_ID
        
        if hasattr(self, 'camera'):
            pass
        else:
            self.camera = pylon.InstantCamera( pylon.TlFactory.GetInstance().CreateFirstDevice() )
            self.camera.Open()
            print("Using device:", self.camera.GetDeviceInfo().GetModelName())

            self.camera.PixelFormat.SetValue(self.pixel_format)  # set pixel depth to 16 bits
            self.camera.ExposureTime.SetValue(self.exposure_time*1000)  # set exposure time 
            self.camera.Width.SetValue(self.size[0])  # set the width
            self.camera.Height.SetValue(self.size[1])  # set the height
            self.camera.OffsetX.SetValue(self.offset[0])  # set the offset x
            self.camera.OffsetY.SetValue(self.offset[1])  # set the offset y
            
            
            # Get the node map
            nodemap = self.camera.GetNodeMap()

            # Set the acquisition frame rate to 60 fps
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(self.sample_rate)
            self.camera.MaxNumBuffer = 15
                    
            # Get the image size
            width = self.camera.Width.GetValue()
            height = self.camera.Height.GetValue()
            self.image_shape = ( (np.arange(height)[::self.subsample]).shape[0], (np.arange(width)[::self.subsample]).shape[0]) 
               
        if start_grabbing:
            self.camera_acq_started = True
        if self.camera_acq_started:
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)  

    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        # Wait for an image and then retrieve it. A timeout of is used.
        #grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image_temp = grabResult.Array[::self.subsample, ::self.subsample]
                current_image_ID = grabResult.GetImageNumber()
                if self.current_image_ID + 1 != current_image_ID:
                    print("Warning: frames might be missed!")
                self.current_image_ID = current_image_ID
                
                grabResult.Release()
                return image_temp.reshape(-1, self.n_channels)
            else:
                return np.empty((0, self.n_channels))
            
        except pylon.TimeoutException:
            return np.empty((0, self.n_channels))
    
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        self.camera.StopGrabbing()
        self.camera.Close()
   
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
        
    def get_data(self, N_points=None):
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        """
        time, data = super().get_data(N_points=N_points)
        data = data.reshape(data.shape[0], self.image_shape[0], self.image_shape[1])
        return time, data
    
    def get_data_PLOT(self):
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        """
        data = super().get_data_PLOT()
        data = data.reshape(data.shape[0], self.image_shape[0], self.image_shape[1])
        return data