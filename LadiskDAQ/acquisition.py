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

    def run_acquisition(self, run_time=None):
        """
        Runs acquisition.
        :params: run_time - (float) number of seconds for which the acquisition will run. 
                 If None acquisition runs indefinitely until self.is_running variable is set
                 False externally (i.e. in a different process)
        """
        self.is_running = True

        self.set_data_source()

        self.plot_data = np.zeros((2, len(self.channel_names)))
        self.set_trigger_instance()

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
        if not os.path.exists(root):
            os.mkdir(root)

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
                       write_start_bytes=None, write_end_bytes=None,
                       channel_names = None, sample_rate=1, timeout=1):
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
        :param: sample_rate - approximate sample rate of incoming signal - currently needed only for plot purposes                   
                           
        """
        super().__init__()

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
    
        self.sample_rate = sample_rate   # TODO: estimate sample_rate  automatically
        self.buffer = b""                # buffer to which recieved data is added

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
            time.sleep(0.3)
        else:
            pass 
        
        # Send commands over serial:
        self.write_to_serial(self.start_bytes_write)
        time.sleep(0.1)

        self.ser.reset_input_buffer() # clears previous data
        self.buffer = b"" # reset class buffer

    def clear_data_source(self):
        time.sleep(0.01)
        self.write_to_serial(self.end_bytes_write)
        time.sleep(0.1)
        self.ser.close()

    def read_data(self):
        # 1) read all data from serial
        self.buffer += self.ser.read_all()
        #print(self.buffer)
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


class SerialAcquisitionSimple(BaseAcquisition):
    """General Purpose Class for Serial Communication where incoming information is
    processed as string
    TODO: fully test this class.
    """
    def __init__(self, port, baudrate, n_channels=1, delim=b",", start_char=b"", end_char=b"\r\n", 
                       write_start_char=None, write_end_char=None,
                       channel_names = None, sample_rate=1, timeout=1,
                       add_time_channel=True):
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
        :param: sample_rate - approximate sample rate of incoming signal - currently needed only for plot purposes                   
                           
        """
        super().__init__()

        self.port = port
        self.baudrate = baudrate
        self.delim = delim
        self.start_char_write = write_start_char
        self.end_char_write   = write_end_char
        self.start_char = start_char
        self.end_char   = end_char
        self.timeout    = timeout

        self.unpack_string = b""
        self.n_channels = n_channels
        self.channel_names = channel_names

        self.set_channel_names()        # sets channel names if none were given to the class
        self.set_data_source()          # initializes serial connection
    
        self.sample_rate = sample_rate   # TODO: estimate sample_rate  automatically
        self.buffer = b""                # buffer to which recieved data is added

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
            time.sleep(0.3)
        else:
            pass 
        
        # Send commands over serial:
        self.write_to_serial(self.start_char_write)
        time.sleep(0.1)

        self.ser.reset_input_buffer() # clears previous data
        self.buffer = b"" # reset class buffer

    def clear_data_source(self):
        time.sleep(0.01)
        self.write_to_serial(self.end_char_write)
        time.sleep(0.1)
        self.ser.close()

    def read_data(self):
        # 1) read all data from serial
        incoming_data = self.ser.read_all()
        #print(incoming_data)
        self.buffer += incoming_data
        #print(self.buffer)
        # 2) split data into lines
        parsed_lines = self.buffer.split(self.end_char + self.start_char)
        if len(parsed_lines) == 1 or len(parsed_lines) == 0: # not enough data
            return np.array([]).reshape(-1, self.n_channels)
        
        # 3) decode full lines, convert data to numpy array
        data = []
        for line in parsed_lines[:-1]: # last element probably does not contain all data
            line_split = [l for l in line.split(self.delim) if len(l) > 0 ]
            if len(line_split) == self.n_channels:
                try:
                    line_decoded = [float(s) for s in line_split if len(s)>0]
                    data.append(line_decoded)
                except:
                    pass
            else:
                #print(f"Expected nr. of bytes {self.expected_number_of_bytes}, line contains {len(line)}")
                pass
        data = np.array(data)
        if len(data) == 0:
            data = data.reshape(-1, self.n_channels)

        # 4) reset buffer with remaninig bytes:
        self.buffer = self.end_char + self.start_char + parsed_lines[-1]

        return data

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
        self.clear_task()

    def read_data(self):
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def set_data_source(self):
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)

