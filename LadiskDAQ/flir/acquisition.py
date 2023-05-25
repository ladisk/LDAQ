import numpy as np
from ctypes import *

from ..acquisition_base import BaseAcquisition

try:
    import PySpin # This will be moved somewhere else in the future
except:
    print("PySpin library not found.")


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
    2) Install PySpin (python wrapper for Spinnaker SDK). On the website listed above, there are
       multiple build wheels listed under "Lastest Windows Python Spinnaker SDK". Choose the one
       that matches your python version and architecture (i.e. spinnaker_python-3.1.0.79-cp310-cp310-win_amd64.zip 
       for python 3.10 64-bit - this is also the version used for development of this class)
    """
    def __init__(self, acquisition_name=None, IRtype='LINEAR_10MK'):
        """
        Args:
            acquisition_name (str, optional): Name of the class. Defaults to None.
            IRtype (str, optional): format type of camera mode. Defaults to 'LINEAR_10MK'. Possible modes include:
                                    - LINEAR_10MK: 10mK temperature resolution
                                    - LINEAR_100MK: 100mK temperature resolution
                                    - RADIOMETRIC: capture radiometric data and manually convert to temperature
                                                    (this requires calibration coefficients, currently some
                                                     calibration values are read from the camera)         
        """
        super().__init__()
        self.acquisition_name = 'FLIR' if acquisition_name is None else acquisition_name
        self.buffer_dtype = np.float16 # this is used when CustomPyTrigger instance is created
        self.virtual_channel_dict = {}
        
        self.image_shape = None # TODO: remove this
        
        self.channel_names_all = []
        self.channel_shapes = []
        self.channel_pos = [] # data position in the buffer (pyTrigger ring buffer)
        
        self.channel_names = []
        self.channel_names_video = []
        
        self.set_IRtype(IRtype) 
        self.camera_acq_started = False
        self.set_data_source()
        
        self.sample_rate = 30 # TODO: this can probably be set in thermal camera and read from it
                              # default camera fps is 30.
        
        # channel in set trigger is actually pixel in flatten array:
        self.set_trigger(1e20, 0, duration=1.0)
        
        # TODO:
        # - set sample rate (either subsample and only acquire every n-th frame or set camera fps)
        # - adjust picture resolution
        # - add new camera source
    
        
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
            pass
        else:
            self.channel_names_all = []
            self.channel_shapes = []
            # Temperature field camera:
            image_shape_temperature = self._init_thermal_camera()
            self.channel_names_all.append('temperature_field')
            self.channel_shapes.append(image_shape_temperature)
            
            # Regular camera:
            # TODO: add regular camera
            
            # add virtual channels:
            for key in self.virtual_channel_dict.keys():
                self.channel_names_all.append(key)
                func, channel_used = self.virtual_channel_dict[key]
                shape_used = self.channel_shapes[ channel_used ]
                dummy_array = np.random.rand( *shape_used )
                output = func(dummy_array)
                
                self.channel_shapes.append( output.shape )
               
            # calculate total number of channels: 
            self.n_channels = len(self.channel_names_all)
            self.n_channels_trigger  = 0
            self.channel_pos = []
            pos = 0
            for shape in self.channel_shapes:
                self.n_channels_trigger += np.prod(shape)
    
                pos_next = pos+np.prod(shape)
                self.channel_pos.append( (pos, pos_next) )
                pos = pos_next
                
            # TODO: this is temporary:
            self.channel_names = [self.channel_names_all[i] for i in range(self.n_channels) if self.channel_shapes[i] == (1,)]
            self.channel_names_video = [self.channel_names_all[i] for i in range(self.n_channels) if self.channel_shapes[i] != (1,)]
                        
            # create buffer for reading all the data in in flatten format:
            self._temp_read_data = np.zeros(self.n_channels_trigger, dtype=self.buffer_dtype)
            
        if not self.camera_acq_started:
            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            self.cam.BeginAcquisition()
            self.camera_acq_started = True

    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        # TODO: add reading of multiple samples
        
        # read thermal camera data:
        data_thermal_camera = self._read_data_thermal_camera() 
        if not data_thermal_camera.shape[0] > 0: # add regular camera condition
            return np.empty((0, self.n_channels_trigger))
        
        i1, i2 = self.channel_pos[0]
        self._temp_read_data[i1:i2] = data_thermal_camera.flatten() # save as flatten
        # read regular camera data:
        # TODO: add regular camera
        
        # read/calculate virtual channels:
        for key in self.virtual_channel_dict.keys():
            func, use_on_channel_idx = self.virtual_channel_dict[key]
            #if use_on_channel_idx == 0: # use on thermal camera:
            #    data_virt_ch = func(data_thermal_camera)
                #i1, i2 = self.channel_pos[ self.channel_names_all.index(key) ]
                #self._temp_read_data[i1:i2] = data_virt_ch.flatten() # save as flatten
                
            data_used_ch = self._temp_read_data[ self.channel_pos[use_on_channel_idx][0] : self.channel_pos[use_on_channel_idx][1] ].reshape(self.channel_shapes[use_on_channel_idx])
            data_virt_ch = func(data_used_ch)    
            i1, i2 = self.channel_pos[ self.channel_names_all.index(key) ]
            self._temp_read_data[i1:i2] = data_virt_ch.flatten() # save as flatten
        
        return self._temp_read_data.reshape(-1, self.n_channels_trigger)
            
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        if self.camera_acq_started:
            #  Image acquisition must be ended when no more images are needed.
            self.cam.EndAcquisition()
            self.camera_acq_started = False
            
        #self._exit_thermal_camera()
   
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
        
    def get_data(self, N_points=None, image=True):
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        
        if image = True, then list of 3D arrays is returned, 
        if False, 2D array is returned with containing virtual channels
        """
        
        time, data = super().get_data(N_points=N_points)
        
        if image:
            channels = [self.channel_names_all.index(name) for name in self.channel_names_video]
            
            data_return = []
            for channel in channels:
                shape = self.channel_shapes[channel]
                pos = self.channel_pos[channel]
                data_return.append( data[:, pos[0]:pos[1]].reshape( (data.shape[0], *shape) ) )
        else:
            channels = [self.channel_names_all.index(name) for name in self.channel_names]
            pos_list = [np.arange(self.channel_pos[channel][0], self.channel_pos[channel][1]) for channel in channels]
            pos_list = np.concatenate(pos_list)
            
            data_return = data[:, pos_list]
    
        return time, data_return
    
    def get_data_PLOT(self, image=False): # this function is actually called only for line plots
        """
        Overwrites the get_data method of the parent class.
        Additionally reshapes the data into a 3D array of shape (n_samples, height, width).
        """
        data = super().get_data_PLOT()
        
        if image:
            channels = [self.channel_names_all.index(name) for name in self.channel_names_video]
            
            data_return = []
            for channel in channels:
                shape = self.channel_shapes[channel]
                pos = self.channel_pos[channel]
                data_return.append( data[:, pos[0]:pos[1]].reshape( (data.shape[0], *shape) ) )
        else:
            channels = [self.channel_names_all.index(name) for name in self.channel_names]
            pos_list = [np.arange(self.channel_pos[channel][0], self.channel_pos[channel][1]) for channel in channels]
            pos_list = np.concatenate(pos_list)
            
            data_return = data[:, pos_list]
        
        return data_return
    
    def _read_data_thermal_camera(self):
        """Reads and retrieves data from the thermal camera.

        Returns:
            np.ndarray: flattened array of pixel values
        """
        image_result = self.cam.GetNextImage()
        #  Ensure image completion
        if image_result.IsIncomplete():
            return  np.empty((0, self.n_channels_trigger))

        # Getting the image data as a np array
        image_data = image_result.GetNDArray()
        if self.CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK:
            # Transforming the data array into a temperature array, if streaming mode is set to TemperatueLinear10mK
            image_Temp_Celsius_high = (image_data *  0.01) - 273.15
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
        
        if image_temp.shape[0] > 0:
            image_result.Release()
        
        return image_temp
        
    def _init_thermal_camera(self):
        ###########################
        #  Thermal Camera Setup   #
        ###########################
        
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()
        # Get current library version
        
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()
        
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

        # Se buffer handling mode:
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
        node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
        
        # set pixel format:
        pixel_format_mono16 = node_pixel_format_mono16.GetValue()
        node_pixel_format.SetIntValue(pixel_format_mono16)
        
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
            image_shape = (self.node_height.GetMax() - self.offsetY.GetMax(), self.node_width.GetValue()-self.offsetX.GetMax())
            return image_shape
        
        except PySpin.SpinnakerException as ex:
            raise Exception('Error: %s' % ex)
        
    def _exit_thermal_camera(self):
        self.cam.DeInit()
        del self.cam
        # Clear camera list before releasing system
        self.cam_list.Clear()
        # Release system instance
        self.system.ReleaseInstance()
        
    def add_virtual_channel(self, virtual_channel_name, channel, function):
        """
        Add a virtual channel to the camera class.
        
        Args:
            virtual_channel_name (str): Name of the channel that will be created
            channel (str): Name or index of the channel on which function will be applied
            function (function): Function used on the image. Takes array shape of the channel as input and has to return a np.array([value]).
        """
        self.terminate_data_source()
        self._exit_thermal_camera()
        if type(channel) == str:
            channel = self.channel_names_all.index(channel)
        self.virtual_channel_dict[virtual_channel_name] = (function, channel)
        self.set_data_source()
    
    