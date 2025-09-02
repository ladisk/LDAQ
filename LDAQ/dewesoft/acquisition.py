import numpy as np
from ctypes import *

from ..acquisition_base import BaseAcquisition
from siriusx import SiriusX


class DewesoftSiriusX(BaseAcquisition):
    """
    Acquisition class for Dewesoft Sirius X devices using OpenDAQ SDK.

    Example:
    ```
    acq = DewesoftSiriusX()
    acq.list_available_devices()
    acq.connect_to_device("daq://Dewesoft_DB24050686")
    channel_settings = {
        0: {
            'Name': 'acc_X',
            'Measurement': 'IEPE', # [IEPE, Voltage]
            'Range': '10000', # [10000, 5000, 1000, 200] mV
            'HPFilter': 'AC 1Hz', # [AC 0.1Hz, AC 1Hz]
            'Excitation': 2.0, # [2, 4, 6] mA
            'Sensitivity': 100,
            'Sensitivity Unit': 'mV/g', # [mV/g, mV/(m/s^2)]
            'Unit': 'g', # [g, m/s^2]
        },
        1: {
            'Name': 'vol_1',
            'Measurement': 'Voltage', # [IEPE, Voltage]
            'Range': '10', # [10, 5, 1, 0.2] V
            'HPFilter': 'DC', # [DC, AC 0.1Hz, AC 1Hz] 
            'Sensitivity': 1,
            'Sensitivity Unit': 'V/V', # [arbitrary]
            'Unit': 'V', # [arbitrary]
        }
    }
    acq.configure_channels(channel_settings, sample_rate)
    ldaq = LDAQ.Core(acquisitions=[acq_sirius])
    ```
    """
    all_acquisitions_ready = False # class property to indicate if all acquisitions are ready to start (not just this one)
    
    def __init__(self, acqusition_name: str = 'dewesoft') -> None:
        """
        EDIT in child class. 
        
        Requirements:
        
        - Make sure to call super().__init__() AT THE BEGGINING of __init__() method.
        
        - Make sure to call self.set_trigger(1e20, 0, duration=1.0) AT THE END (used just for inititialization of buffer).
        """
        super().__init__()
        self.sirius = SiriusX()
        self.acquisition_name = acqusition_name
        self.set_trigger(1e20, 0, duration=1.0)

    def list_available_devices(self):
        """
        This method will list all the available devices that are found by the 
        OpenDAQ SDK.
        """
        self.sirius.list_available_devices()

    def connect_to_device(self, connection_string: str):
        """
        This method will connect to a specific device using the provided 
        connection string.

        Notice:
        OpenDAQ for now only supports Etherenet connection to the DewesoftX 
        device. Also, for the device to accept the connection, clients (your) 
        subnet must be the same as the device's subnet. Default SiriusX IP is
        192.168.10.1.

        Parameters
        ----------
        connection_string : str
            The connection string used to connect to the device.
            Example: "daq://Dewesoft_DB24050686"

        Returns
        -------
        bool
            True if the connection was successful, False otherwise.
        """
        return self.sirius.connect(connection_string)

    def configure_channels(self, channel_settings: dict, sample_rate: int):
        """
        Configure multiple channels.

        Parameters
        ----------
        channel_settings : dict
            A dictionary containing the settings for each channel.
            Example:
            ```
            channel_settings = {
                0: {
                    'Name': 'acc_X',
                    'Measurement': 'IEPE', # [IEPE, Voltage]
                    'Range': 10000, # [10000, 5000, 1000, 200] mV
                    'HPFilter': 'AC 1Hz', # [AC 0.1Hz, AC 1Hz]
                    'Excitation': 2.0, # [2, 4, 6] mA
                    'Sensitivity': 100,
                    'Sensitivity Unit': 'mV/g', # [mV/g, mV/(m/s^2)]
                    'Unit': 'g', # [g, m/s^2]
                },
                1: {
                    'Name': 'vol_1',
                    'Measurement': 'Voltage', # [IEPE, Voltage]
                    'Range': 10, # [10, 5, 1, 0.2] V
                    'HPFilter': 'DC', # [DC, AC 0.1Hz, AC 1Hz] 
                    'Sensitivity': 1,
                    'Sensitivity Unit': 'V/V', # [arbitrary]
                    'Unit': 'V', # [arbitrary]
                }
            }            
            ```
            Keys must be integers representing channel numbers of the Sirius X 
            device.
        sample_rate : int
            The sample rate to be used for the acquisition in samples/second.
        """
        # configuring channels
        self.sirius.configure_channels(channel_settings)
        print("Channels configured. You can check the current configuration "
              "with self.list_available_channels()")

        # getting the channel names as strings from the channel settings
        ch_names = [ch['Name'] for ch in channel_settings.values()]
        self._channel_names_init = ch_names

        # setting the sample rate
        self.sample_rate = self.sirius.set_sample_rate(int(sample_rate))
        print(f"Sample rate was set to: {self.sample_rate} Hz.")

        # call set all channels to prepare info for trigger
        self._set_all_channels()
    
    def list_available_channels(self):
        """
        This method will list all the available channels with configured
        settings.
        """
        self.sirius.list_available_channels()

    def set_data_source(self) -> None:
        """EDIT in child class.
        
        Properly sets acquisition source before measurement is started. Requirements for this method:
        
         - Should call super().set_data_source() AT THE END of the method.
         
         - Should be set up in a way that it is able to be called multiple times in a row without issues.  
         
         - Should set up self._channel_names_init and self._channel_names_video_init if not set in __init__() method.
         
        VIDEO source only: 
         - Should set self._channel_shapes_video_init which is a list of tuples with shapes of each video channel that will be recieved from acquisition source. This is required for proper operation of the class. 
         
         - the order of the shapes in self._channel_shapes_video_init should be the same as the order of the channels in self._channel_names_video_init.
        """
        self.sirius.create_reader()
        self.sirius.start_reader()
        super().set_data_source()
    
    def terminate_data_source(self)->None:
        """EDIT in child class.
        
        Properly closes/disconnects acquisition source after the measurement. The method should be able to handle mutliple calls in a row.
        """
        self.sirius.stop_reader()

    def read_data(self)->np.ndarray:
        """EDIT in child class.
        
        This method only reads data from the source and transforms data into 
        standard format used by other methods.
        It is called within self.acquire() method which properly handles 
        acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must ALWAYS return a 2D numpy array of shape (n_samples, n_columns).

        Returns:
            data (np.ndarray): 2D numpy array of shape (n_samples, n_columns)
        """
        samples_to_read = self.sirius.available_samples()
        timeout = 0.1 # sec
        data = self.sirius.read_processed(
            sample_count=samples_to_read, timeout=timeout)
        
        if data.size == 0:
            data = np.empty((0, len(self._channel_names_init)))

        return data

    def get_sample_rate(self)->float:
        """EDIT in child class (Optional).
        
        Returns sample rate of acquisition class.
        """
        return self.sample_rate
    
    def clear_buffer(self)->None:
        """EDIT in child class (Optional).
        
        The source buffer should be cleared with this method. It can either clear the buffer, or
        just read the data with self.read_data() and does not add/save data anywhere. By default, this method
        will read the data from the source and not add/save data anywhere.
        
        Returns None.
        """
        self.read_data()
            


