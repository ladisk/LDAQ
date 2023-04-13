import os
import numpy as np
import pickle
import datetime
import time
import serial
import struct

from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.Task import Task
from pyTrigger import pyTrigger
from .daqtask import DAQTask
import threading

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
    time_global = time.time()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_acquired_samples  = 0  # number of samples acquired from run
        self.N_new_samples = 0
        self.time_triggered = time.time()
        self.N_triggers = 0
        
        self.first_trigger = True

    def _add_data_to_buffer(self, data):
        rows_left_before = self.rows_left
        super()._add_data_to_buffer(data)
        N = rows_left_before - self.rows_left
        
        self.N_acquired_samples += data.shape[0]
        self.N_new_samples      += N
        
        if N != data.shape[0] and N != 0:
            print("rows_left", rows_left_before, "N", N, "data", data.shape[0])
        #if N != 0:
        #    print("N:", N)
        
    def _add_data_chunk(self, data):
        super()._add_data_chunk(data)
        if self.triggered and self.first_trigger:
            CustomPyTrigger.triggered_global = True 
            print()
            print("LOCAL")
        elif CustomPyTrigger.triggered_global and self.first_trigger:
            self.triggered = True
            print()
            print("GLOBAL")
        else:
            pass

        if self.first_trigger and (self.triggered or CustomPyTrigger.triggered_global):
            self.time_triggered     = time.time()
            self.N_triggers        += 1
            self.first_trigger      = False
        return 
        
    def add_data(self, data):
        finished = super().add_data(data)  
        return finished
    
    def get_data_new(self):
        """Retrieved any new data after trigger event that has been not yet retrieved.

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
        else:
            print()
            print("Reading before trigger.")
            return np.empty(shape=(0, self.ringbuff.columns))
    
    def _trigger_index(self, data):
        trigger = super()._trigger_index(data)
        if type(trigger) != np.ndarray:
            self.N_new_samples += self.presamples - trigger
        return trigger

    
class BaseAcquisition:
    all_acquisitions_ready = False
    
    def __init__(self):
        """EDIT in child class"""
        self.acquisition_name = "DefaultAcquisition"
        self.channel_names = []
        self.is_running = True
        self.is_standalone = True # if this is part of bigger system or used as standalone object
        self.is_ready = False
    
        self.lock_acquisition = threading.Lock() # ensures acquisition class runs properly if used in multiple threads.
        
        # child class needs to have variables below:
        self.n_channels  = 0
        self.sample_rate = 0
        
        CustomPyTrigger.time_global = time.time()
        
    def read_data(self):
        """EDIT in child class
        
        This code acquires data. 
        
        Must return a 2D array of shape (n_samples, n_columns).
        """
        pass

    def terminate_data_source(self):
        """EDIT in child class"""
        pass

    def set_data_source(self):
        """EDIT in child class"""
        pass
    
    def get_sample_rate(self):
        """EDIT in child class
        
        This function is also useful to compute sample_rate estimation if no sample rate is given
        """
        return self.sample_rate
    
    def clear_buffer(self):
        """EDIT in child class
        
        The source buffer should be cleared (all data deleted) with this method.
        """

    # The following methods should work without changing.
    def stop(self):
        self.is_running = False
    
    def acquire(self):
        with self.lock_acquisition: # lock to secure variables
            acquired_data = self.read_data()
            self.Trigger.add_data(acquired_data)
           
        if self.Trigger.finished or not self.is_running:            
            self.stop()
            self.terminate_data_source()
            

    def get_data(self, N_points=None):
        """Reads and returns data from the pyTrigger buffer.
        :param N_points: number of last N points to read from pyTrigger buffer. 
                            if N_points="new", then only new points will be retrieved.
                            if None whole buffer is returned.
        Returns:
            tuple: (time, data) - 1D time vector and 2D measured data, both np.ndarray
        """
        
        if N_points is None:
            data = self.Trigger.get_data()#N_points=self.Trigger.N_acquired_samples)
            
        elif N_points == "new":
            with self.lock_acquisition:
                data = self.Trigger.get_data_new()
        else:
            data = self.Trigger.get_data()[-N_points:]
                
        time = np.arange(data.shape[0])/self.sample_rate     
        return time, data
    
    def get_measurement_dict(self, N_points=None):
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

        self.set_trigger_instance()
        self.set_data_source()
        
        # if acquisition is used in some other classes, wait until all acquisition sources are ready:
        if not self.is_standalone:
            self.is_ready = True    # this source is ready (other may not be)
            while not BaseAcquisition.all_acquisitions_ready: # until every source is ready
                self.clear_buffer()                           # reads data, does not store in anywhere
                #self.acquire()
                time.sleep(0.01)
                if not self.is_running:
                    break
        
        self.run_time = 0 # actual time run to obtain all samples
        time_start_acq = time.time()
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
        time_end_acq  = time.time()
        self.run_time = time_end_acq-time_start_acq

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
        
        self.set_trigger_instance()
        
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
            
        #TODO: some parameters in dict - duration_smaples, duration_seconds are not updated
        self.set_trigger_instance()
        
    def set_trigger_instance(self):
        self.Trigger = CustomPyTrigger( #pyTrigger
        rows=self.trigger_settings['duration_samples'], 
        channels=self.n_channels,
        trigger_type=self.trigger_settings['type'],
        trigger_channel=self.trigger_settings['channel'], 
        trigger_level=self.trigger_settings['level'],
        presamples=self.trigger_settings['presamples'])
           
    def activate_trigger(self, all_sources=True):
        """Sets trigger off. Useful if the acquisition class is trigered by another process.
            This trigger can also trigger other acquisition sources by setting property class
        """
        if all_sources:
            CustomPyTrigger.triggered_global = True

    def reset_trigger(self):
        CustomPyTrigger.triggered_global = False
        self.Trigger.triggered = False
        
    def is_triggered(self):
        """Checks if acquisition class has been triggered during measurement.

        Returns:
            bool: True/False if triggered
        """
        return self.Trigger.triggered
        
    def _all_acquisitions_ready(self):
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


class ADAcquisition(BaseAcquisition):
    def __init__(self, port_nr):
        super.__init__()


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

        self.task_name = task_name
        self.acquisition_name = task_name if acquisition_name is None else acquisition_name

        try:
            DAQmxClearTask(taskHandle_acquisition)
        except:
            pass

        self.Task = DAQTask(self.task_name)
        glob_vars = globals()
        glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        self.channel_names = self.Task.channel_list
        self.sample_rate = self.Task.sample_rate
        self.n_channels = self.Task.number_of_ch

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def clear_task(self):
        """Clear a task."""
        self.Task.clear_task(wait_until_done=False)
        del self.Task

    def terminate_data_source(self):
        self.clear_task()

    def read_data(self):
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def clear_buffer(self):
        self.Task.acquire_base()
    
    def set_data_source(self):
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)

