import numpy as np
import time
import sys

# Analog Discovery 2:
from . import dwfconstants as dwfc 
from ctypes import *

from ..acquisition_base import BaseAcquisition


class WaveFormsAcquisition(BaseAcquisition):
    def __init__(self, channels=[0, 1], sample_rate=10000, 
                 channel_names=None, acquisition_name=None, device_number=None):
        super().__init__()

        self.acquisition_name = 'AD2' if acquisition_name is None else acquisition_name
        self._channel_names_init = channel_names if channel_names is not None else [f'CH{i}' for i in channels]
        
        
        self.channel_idx = channels
        self.sample_rate = sample_rate
        self.device_number = device_number if device_number is not None else -1
        
        if sys.platform.startswith("win"):
            self.dwf = cdll.dwf
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

        self.hdwf = c_int(0) # device handle
        
        # tracking of lost data and corrupted data:
        self.cLost = c_int()
        self.cCorrupted = c_int()
        self.fLost = 0
        self.fCorrupted = 0
        
        self.configure_channels() # configure channel range
        
        self.set_data_source()
        self.set_trigger(1e20, 0, duration=1.0)
        
        
    def configure_channels(self, input_range=None):
        """Specify min and max value range for each channel.
        Args:
        input_range (dict): dictionary with channel index as key and tuple of min and max values as value. channel indices
                            have to be the same as self.channel_idx (or channels input parameters in init)
                            For example: {0:(-10, 10), 1:(-5, 5)} 
                            -> channel 0 has range -10 to 10 V and channel 1 has range -5 to 5 V.
                            
        """
        if input_range is None:
            if not hasattr(self, 'input_range'):
                input_range = {idx:(-10, 10) for idx in self.channel_idx}
                self.input_range = input_range
                    
        # based on which channels are used:
        for idx in self.channel_idx:
            val_min, val_max = self.input_range[idx]
            ch_range = val_max - val_min
            ch_offset = (val_max + val_min)/2
            
            # enable channel:
            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(idx), c_bool(True))
            # set range:
            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(idx), c_double(ch_range))
            # set offset:
            self.dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, c_int(idx), c_double(ch_offset))
        
        
    def read_data(self):
        """
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        sts = c_byte() # acquisition status
        cAvailable = c_int() # number of samples available
        
        self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))
        if (sts == dwfc.DwfStateConfig or sts == dwfc.DwfStatePrefill or sts == dwfc.DwfStateArmed) :
            # Acquisition not yet started.
            return np.empty((0, self.n_channels))

        self.dwf.FDwfAnalogInStatusRecord(self.hdwf, byref(cAvailable), byref(self.cLost), byref(self.cCorrupted))
        if self.cLost.value :
            self.fLost = 1
        if self.cCorrupted.value :
            self.fCorrupted = 1

        if cAvailable.value==0: # no data available
            return np.empty((0, self.n_channels))
        
        arr = []
        for i in self.channel_idx:
            rgdSamples = (c_double*cAvailable.value)()
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(i), byref(rgdSamples), cAvailable) # get channel 1 data
            values = np.fromiter(rgdSamples, dtype =float)
            arr.append(values)
        arr = np.array(arr).T
        return arr

   
    def set_data_source(self):
        """
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        
        if self.hdwf.value == dwfc.hdwfNone.value: # if device is not open
            self.dwf.FDwfDeviceOpen(self.device_number, byref(self.hdwf))

        self.dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, dwfc.acqmodeRecord)
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.sample_rate))
        self.dwf.FDwfAnalogInRecordLengthSet(self.hdwf, c_double(-1)) # -1 infinite record length
        
        self.configure_channels()
        #wait at least 2 seconds for the offset to stabilize
        time.sleep(0.3)

        # check if the device is running:
        device_state = c_int()
        self.dwf.FDwfAnalogInStatus(self.hdwf, c_bool(True), byref(device_state)) 
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
            
        super().set_data_source()
     
    
    def terminate_data_source(self):
        """        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        #self.dwf.FDwfAnalogOutReset(self.hdwf, c_int(0))
        self.dwf.FDwfDeviceCloseAll()
        self.hdwf = c_int(0)
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
        self.read_data()