import numpy as np
import time
from ctypes import *

# Serial communication:
import serial
import struct

from ..acquisition_base import BaseAcquisition

    
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
    