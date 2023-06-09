import numpy as np
import time

from ..acquisition_base import BaseAcquisition

class SimulatedAcquisition(BaseAcquisition):
    """
    Simulated acquisition class that can be used when no source is present.
    """
    def __init__(self):
        """
        
        """
        super().__init__()
        
        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source
        
        
        self.set_trigger(1e20, 0, duration=1.0)
        
    def set_simulated_data(self, data, channel_names=None):
        """sets simulated data to be returned by read_data() method. 
        This should also update self._channel_names_init list.

        Args:
            data (np.ndarray): numpy array with shape (n_samples, n_channels)
        """
         
    def set_simulated_video(self, video, channel_names_video=None):
        """sets simulated video to be returned by read_data() method.
        This should also update self._channel_names_video_init and self._channel_shapes_video_init lists.

        Args:
            data (list): list of numpy arrays with shape (n_samples, width, height), where each elements
                         in list corresponds to one video channel.
        """
        
    def set_data_source(self):
        """
        Initializes simulated data source
        """
        
        
        super().set_data_source()
        
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
        
        data = 0

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
    