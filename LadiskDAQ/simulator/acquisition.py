import numpy as np
import time

from ..acquisition_base import BaseAcquisition

class SimulatedAcquisition(BaseAcquisition):
    """
    Simulated acquisition class that can be used when no source is present.
    """
    def __init__(self, acquisition_name=None):
        """
        Args:
            acquisition_name (str, optional): Name of the acquisition. Defaults to None, in which case the name "Simulator" is used.
        """
        super().__init__()
        
        self.acquisition_name = 'Simulator' if acquisition_name is None else acquisition_name

        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source
        
        
        self.set_trigger(1e20, 0, duration=1.0)
        
    def set_simulated_data(self, fun, channel_names=None, sample_rate=None, args=()):
        """sets simulated data to be returned by read_data() method. 
        This should also update self._channel_names_init list.

        Args:
            fun (function): function that returns numpy array with shape (n_samples, n_channels)
            channel_names (list, optional): list of channel names. Defaults to None, in which case the names "channel_0", "channel_1", ... are used.
            sample_rate (int, optional): sample rate of the simulated data. Defaults to None, in which case the sample rate of 1000 Hz is used.
            args (tuple, optional): arguments for the function. Defaults to ().
        """
        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source

        self.simulated_function = fun
        self._channel_names_init = channel_names
        self.sample_rate = 1000 if sample_rate is None else sample_rate
        self._args = args
        
        time_array = np.arange(self.sample_rate)/self.sample_rate
        data = fun(time_array, *self._args)

        if data.ndim == 2:

            if channel_names is None:
                self._channel_names_init = [f"channel_{i}" for i in range(data.shape[1])]

            if data.shape[1] != len(self._channel_names_init):
                raise ValueError("Number of channels in data and channel_names does not match.")
        else:
            raise ValueError("Data must be 2D array.")

    def set_simulated_video(self, fun, channel_name_video=None, sample_rate=None, args=()):
        """sets simulated video to be returned by read_data() method.
        This should also update self._channel_names_video_init and self._channel_shapes_video_init lists.

        Args:
            fun (function): function that returns numpy array with shape (n_samples, width, height)
            channel_name_video (str, optional): name of the video channel. Defaults to None, in which case the name "video" is used.
            sample_rate (int, optional): sample rate of the simulated data. Defaults to None, in which case the sample rate of 30 Hz is used.
            args (tuple, optional): arguments for the function. Defaults to ().
        """
        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source

        self.simulated_function = fun
        self.sample_rate = 30 if sample_rate is None else sample_rate
        self._args = args
        
        time_array = np.arange(self.sample_rate)/self.sample_rate
        data = fun(time_array, *self._args)

        if data.ndim == 3:
            if channel_name_video is None:
                self._channel_names_video_init = ["video_channel"]
            else:
                self._channel_names_video_init = [channel_name_video]

            self._channel_shapes_video_init = [data.shape[1:]]
        else:
            raise ValueError("Data must be 3D array.")

        
    def set_data_source(self):
        """
        Initializes simulated data source
        """
        self.time_start = time.time()
        self.time_previous = self.time_start
        self.time_add = 0

        super().set_data_source()

        self.set_trigger(1e20, 0, duration=1.0)
        
    def terminate_data_source(self):
        """
        Terminates simulated data source
        """
        pass

    def read_data(self):
        """reads data from simulated data source.

        Returns:
            np.ndarray: data from serial port with shape (n_samples, n_channels).
        """
        time.sleep(0.1)
        
        time_now = time.time()
        time_elapsed = time_now - self.time_previous
        samples_to_read = int(time_elapsed * self.sample_rate)
        time_array = np.arange(samples_to_read)/self.sample_rate + self.time_add

        data = self.simulated_function(time_array, *self._args)

        if data.ndim == 3:
            data = data.reshape((-1, data.shape[1]*data.shape[2]))

        self.time_previous = time_now
        self.time_add = time_array[-1] + 1/self.sample_rate
        return data
    
    def clear_buffer(self):
        """
        Clears serial buffer.
        """
        self.read_data()
            
    def get_sample_rate(self):
        """Returns acquisition sample rate.

        Returns:
            float: estimated sample rate
        """
        
        return self.sample_rate
    