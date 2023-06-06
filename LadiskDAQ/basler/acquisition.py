import numpy as np
from ctypes import *

from ..acquisition_base import BaseAcquisition
    
try:    
    from pypylon import pylon
except:
    pass



class BaslerCamera(BaseAcquisition):
    """
    Acquisition class for Basler camera using pypylon library.
    
    Link to required programs:
    https://www.baslerweb.com/en/downloads/software-downloads/#type=pylonsoftware;language=all;version=7.3.0
    
    Installation steps:
    1) Download and install pylon 7.3.0 Camera Software Suite Windows software and choose developer option during installation
    2) Install python library with pip install pypylon
    
    """
    def __init__(self, acquisition_name=None, sample_rate=10, channel_name_camera="Camera", offset=(0, 0), size=(3000, 2000),
                 subsample=1, pixel_format="Mono12", exposure_time_ms=4.0):
        """Initializes acquisition class for Basler camera.

        Args:
            acquisition_name (str, optional):    acquisition name. Defaults to None in which case 'BaslerCamera' is default name.
            sample_rate (int, optional):         camera sample rate. Can be set up to 60 fps. Defaults to 10.
            channel_name_camera (str, optional): name of the default video channel. Defaults to "Camera".
            offset (tuple, optional):            Pixel offset (X, Y). Defaults to (0, 0).
            size (tuple, optional):              Image size from the offset (width, height). Defaults to (3000, 2000).
            subsample (int, optional):           Adjust resolution of basler camera by subsampling retrieved image. Defaults to 1.
                                                 TODO: this is a software subsampling, not hardware subsampling, in the future this
                                                 should be set in the camera itself.
            pixel_format (str, optional):        Format of each. Defaults to "Mono12".
            exposure_time_ms (float, optional):  exposure time in milliseconds. Defaults to 4.0.
        """
        try:
            pylon # check if pylon is imported
        except:
            raise Exception("Pypylon library not found. Please install it before using this class.")

        super().__init__()

        self.acquisition_name = 'BaslerCamera' if acquisition_name is None else acquisition_name
        
        self.sample_rate = sample_rate # camera fps
        self.subsample = subsample     # subsample factor to reduce resolution # TODO: should be set in camera
        self.size = size     # camera size
        self.offset = offset # camera offsets
        self.pixel_format = pixel_format
        self.exposure_time = exposure_time_ms # in ms
        
        self._channel_names_video_init = [channel_name_camera]
        self._channel_shapes_video_init = [] # in this case, this is set in set_data_source() method
        
        self.buffer_dtype = np.uint16    # TODO: adjust this according to pixel_format
        self.camera_acq_started = False  # flag, where True means that camera is acquiring data
        
        self.set_data_source(start_grabbing=False) # initialize camera, get channel shapes
        self.set_trigger(1e20, 0, duration=1.0)    
       
    def set_data_source(self, start_grabbing=True):
        """
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        # TODO: support for multiple cameras
        
        self.current_image_ID = 0 # reset image_ID
        if hasattr(self, 'camera'):
            pass
        else:
            self.camera = pylon.InstantCamera( pylon.TlFactory.GetInstance().CreateFirstDevice() )
            self.camera.Open()
            #print("Using device:", self.camera.GetDeviceInfo().GetModelName())

            self.camera.PixelFormat.SetValue(self.pixel_format)         # set pixel depth 
            self.camera.ExposureTime.SetValue(self.exposure_time*1000)  # set exposure time 
            self.camera.Width.SetValue(self.size[0])      # set the width
            self.camera.Height.SetValue(self.size[1])     # set the height
            self.camera.OffsetX.SetValue(self.offset[0])  # set the offset x
            self.camera.OffsetY.SetValue(self.offset[1])  # set the offset y
            
            # Get the node map
            nodemap = self.camera.GetNodeMap()

            # Set the acquisition frame rate to self.sample_rate
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(self.sample_rate)
            self.sample_rate = self.camera.AcquisitionFrameRate.GetValue() # get actual sample rate
            self.camera.MaxNumBuffer = 15
                    
            # Get the image size
            width = self.camera.Width.GetValue()
            height = self.camera.Height.GetValue()
            
            # save video channel shape
            image_shape = ( (np.arange(height)[::self.subsample]).shape[0], (np.arange(width)[::self.subsample]).shape[0])
            self._channel_shapes_video_init.append(image_shape)
               
        if start_grabbing:
            if not self.camera_acq_started:
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.camera_acq_started = True
              
        super().set_data_source() # call parent method to set all channel names and shapes

    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        # TODO: do this for each source camera when multiple camers will be supported:
        # TODO: allow for retrieving multiple images at once
        
        # get camera shape:
        shape = self.channel_shapes[ self.channel_names_all.index(self.channel_names_video[0]) ] # 1st video channel
        N_pixels = shape[0]*shape[1]
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image_temp = grabResult.Array[::self.subsample, ::self.subsample]
                current_image_ID = grabResult.GetImageNumber()
                if self.current_image_ID + 1 != current_image_ID:
                    print("Warning: frames might be missed!")
                self.current_image_ID = current_image_ID
                
                grabResult.Release()
                
                self._temp_image = image_temp
                return image_temp.reshape(-1, N_pixels)
            else:
                return np.empty((0, N_pixels))
            
        except pylon.TimeoutException:
            return np.empty((0, N_pixels))
    
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        if hasattr(self, 'camera'):
            if self.camera_acq_started:
                self.camera.StopGrabbing()
                self.camera_acq_started = False
            self.camera.Close()
        else:
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
        self.read_data() # TODO: maybe camera has some built-in function to clear its buffer?
        
  
    