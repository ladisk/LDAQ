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


class BaseAcquisition:
    def __init__(self):
        """EDIT in child class"""
        self.channel_names = []
        self.plot_data = []
        self.is_running = True

        # child class needs to have variables below:
        self.n_channels = 0
        self.sample_rate = 0

    def read_data(self):
        """EDIT in child class
        
        This code acquires data. 
        
        Must return a 2D array of shape (n_samples, n_columns).
        """
        pass

    def clear_data_source(self):
        """EDIT in child class"""
        pass

    def set_data_source(self):
        """EDIT in child class"""
        pass

    # The following methods should work without changing.
    def stop(self):
        self.is_running = False
    
    def acquire(self):
        acquired_data = self.read_data()
        self.plot_data = np.vstack((self.plot_data, acquired_data))
        self.Trigger.add_data(acquired_data)
            
        if self.Trigger.finished or not self.is_running:
            self.data = self.Trigger.get_data()

            self.stop()
            self.clear_data_source()

    def run_acquisition(self):
        self.is_running = True

        self.set_data_source()

        self.plot_data = np.zeros((2, len(self.channel_names)))
        self.set_trigger_instance()

        while self.is_running:
            time.sleep(0.01)
            self.acquire()

    def set_trigger(self, level, channel, duration=1, duration_unit='seconds', presamples=100, type='abs'):
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

    def set_trigger_instance(self):
        self.Trigger = pyTrigger(
            rows=self.trigger_settings['duration_samples'], 
            channels=self.n_channels,
            trigger_type=self.trigger_settings['type'],
            trigger_channel=self.trigger_settings['channel'], 
            trigger_level=self.trigger_settings['level'],
            presamples=self.trigger_settings['presamples'])
    
    def save(self, name, root='', save_channels='All', timestamp=True, comment=''):
        """Save acquired data.
        
        :param name: filename
        :param root: directory to save to
        :param save_channels: channel indices that are save. Defaults to 'All'.
        :param timestamp: include timestamp before 'filename'
        :param comment: commentary on the saved file
        """
        self.data_dict = {
            'data': self.data,
            'channel_names': self.channel_names,
            'comment': comment,
        }

        if hasattr(self, 'sample_rate'):
            self.data_dict['sample_rate'] = self.sample_rate

        if save_channels != 'All':
            self.data_dict['data'] = np.array(self.data_dict['data'])[:, save_channels]
            self.data_dict['channel_names'] = [_ for i, _ in enumerate(self.data_dict['channel_names']) if i in save_channels]

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        self.filename = f'{stamp}{name}.pkl'
        self.path = os.path.join(root, self.filename)
        pickle.dump(self.data_dict, open(self.path, 'wb'), protocol=-1)


class ADAcquisition(BaseAcquisition):
    def __init__(self, port_nr):
        super.__init__()


class SerialAcquisition(BaseAcquisition):
    """General Purpose Class for Serial Communication"""
    def __init__(self, port, baudrate, byte_sequence, start_bytes=b"\xfa\xfb", end_bytes=b"\n", 
                       write_bytes=None, channel_names = None, sample_rate=1):
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
        :param: write_bytes - (tuple) sequence of b"" strings with bytes to write to initiate/set data transfer or other settings
                              Writes each encoded bstring with 1ms delay.
        :param: sample_rate - approximate sample rate of incoming signal - currently needed only for plot purposes
                                
                            
        """
        super().__init__()

        self.port = port
        self.baudrate = baudrate
        self.byte_sequence = byte_sequence
        self.write_bytes = write_bytes
        self.start_bytes = start_bytes
        self.end_bytes = end_bytes

        self.unpack_string = b""
        self.expected_number_of_bytes = 0
        self.n_channels = 0
        self.channel_names = channel_names

        self.set_unpack_data_settings() # sets unpack_string, expected_number_of_bytes, n_channels
        self.set_channel_names()        # sets channel names if none were given to the class
        self.set_data_source()          # initializes serial connection
    
        self.sample_rate = sample_rate   # TODO: estimate sample_rate  automatically
        self.buffer = b""                # buffer to which recieved data is added

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def clear_data_source(self):
        return self.ser.close()

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
    
    def set_data_source(self):
        if not hasattr(self, 'ser'):
            try:
                self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)
            except serial.SerialException:
                print("Serial port is in use.")
        elif not self.ser.is_open:
            self.ser.open()
        else:
            pass 

        if self.write_bytes is None:
            pass
        else:
            #TODO: write self.write_bytes sequence
            pass

        self.ser.read_all() # clears previous data
        self.ser.read_all()
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


class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class."""

    def __init__(self, task_name):
        """Initialize the task.

        :param task_name: Name of the task from NI Max
        """
        super().__init__()

        self.task_name = task_name

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

    def clear_data_source(self):
        return self.clear_task()

    def read_data(self):
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def set_data_source(self):
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)

