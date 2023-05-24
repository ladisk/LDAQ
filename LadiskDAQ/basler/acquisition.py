import numpy as np
from ctypes import *

from ..acquisition_base import BaseAcquisition
    
try:    
    from pypylon import pylon
except:
    print("pypylon library not found. Please install using pip install pypylon")



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
    