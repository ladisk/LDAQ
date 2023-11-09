import os
import numpy as np
import pickle
import datetime
import time
import threading
import inspect

from pyTrigger import pyTrigger, RingBuffer2D
from ctypes import *


class CustomPyTrigger(pyTrigger):
    """
    Upgrades pyTrigger class with features needed for acquisition class BaseAcquisition.
    """
    triggered_global = False
    def __init__(self, rows:int=5120, channels:int=4, trigger_channel:int=0,
                 trigger_level:float=1., trigger_type:str='up', presamples:int=1000,
                 dtype:np.dtype=np.float64)->None:      
        """
        Upgrades pyTrigger class with features needed for acquisition class BaseAcquisition.
        This class handles:
        
        - creating ring buffer for storring measured data, 
        - triggering acquisition,
        - tracking number of acquired samples,
        - retrieving data from ring buffer,
        
        Args:
            rows (int): number of rows
            channels (int): number of channels in ring buffer that will be created
            trigger_channel (int): the channel in ring buffer used for triggering
            trigger_level (float): the level to cross, to start trigger
            trigger_type (str): 'up' is default, possible also 'down'/'abs'
            presamples (int): number of presamples
            dtype (numpy.dtype): dtype of the data
        """
        self.rows = rows
        self.channels = channels
        self.trigger_channel = trigger_channel
        self.trigger_level = trigger_level
        self.trigger_type = trigger_type.lower()
        self.presamples = presamples
        self.ringbuff =  RingBuffer2D(rows=self.rows, columns=self.channels, dtype=dtype)
        self.triggered = False
        self.rows_left = self.rows
        self.finished = False
        self.first_data = True
        
        self.N_acquired_samples               = 0 # samples acquired throughout whole acquisition process
        self.N_acquired_samples_since_trigger = 0 # samples acquired since trigger
        self.N_new_samples                    = 0 # new samples that have not been retrieved yet
        self.N_new_samples_PLOT               = 0 # new samples that have not been retrieved yet - Plotting purposes
        self.N_triggers                       = 0 # amount of time acquisition was triggered (should be 1 at the end of the measurement)
        
        self.first_trigger = True
        
        self.continuous_mode = False # continuous acquisition without definite stop.
        self.N_samples_to_acquire = self.rows # amount of samples to acquire in continuous mode.

    def _add_data_to_buffer(self, data:np.ndarray)->None:
        """Upgrades parent _add_data_to_buffer() to track sample variables
           N_acquired_samples, N_new_samples, N_acquired_samples_since_trigger
        """
        if self.continuous_mode:
            if self.triggered and not (self.rows_left > len(data)):
                # if data is larger then rows_left and if continuous mode is enabled,
                # save all data, reset buffer and continue with acquisition
                self.reset_buffer()
                
            if self.N_samples_to_acquire is not None: # if measurement duration is specified
                if self.N_acquired_samples_since_trigger + len(data) >= self.N_samples_to_acquire:
                    data = data[:self.N_samples_to_acquire - self.N_acquired_samples_since_trigger]
                    self.finished = True
            
        rows_left_before = self.rows_left
        super()._add_data_to_buffer(data)
        N = rows_left_before - self.rows_left
        
        self.N_acquired_samples += data.shape[0]
        self.N_new_samples_PLOT += data.shape[0]
        self.N_new_samples      += N
        self.N_acquired_samples_since_trigger += N
        
    def _add_data_chunk(self, data:np.ndarray)->None:
        """Upgrades parent _add_data_chunk() to globally trigger all acquisition sources present
           in the measurement process, or that another acquisition source triggers this class.
           Global trigger is implemented via class property variable 'triggered_global'.
        """
        super()._add_data_chunk(data)
        if self.triggered and self.first_trigger:
            CustomPyTrigger.triggered_global = True 
        elif CustomPyTrigger.triggered_global and self.first_trigger:
            self.triggered = True
        else:
            pass

        if self.first_trigger and (self.triggered or CustomPyTrigger.triggered_global):
            self.N_triggers    += 1
            self.first_trigger  = False
        return 
    
    def get_data_new(self)->np.ndarray:
        """Retrieves any new data from ring buffer, stored AFTER trigger event, that has been not yet retrieved.

        Returns:
            np.ndarray: data of shape (rows, channels)
        """
        if self.triggered:
            data = self.ringbuff.get_data()
            if self.N_new_samples > 0:
                data = data[-self.N_new_samples:]
            else:
                data = np.empty(shape=(0, self.channels))
            self.N_new_samples = 0
            
            return data
        else: # NOTE: this should not happen!
            return np.empty(shape=(0, self.ringbuff.columns))
        
    def get_data_new_PLOT(self)->np.ndarray:
        """Retrieves any new data from ring buffer that has been not yet retrieved. 
           This method should be used for plotting purposes only, as it has separate
           samples tracking variable N_new_samples_PLOT.

        Returns:
            np.ndarray: data of shape (rows, channels)
        """
        if self.N_new_samples_PLOT > 0:
            data = self.ringbuff.get_data()[-self.N_new_samples_PLOT:]
            self.N_new_samples_PLOT = 0
            return data
        else:
            return np.empty(shape=(0, self.channels))
    
    def _trigger_index(self, data:np.ndarray)->int|np.ndarray:
        """Upgrades parent _trigger_index() method. Beside searching for trigger event, it
           adds amount of samples missed by _add_data_to_buffer() in case of use of presamples.
        """
        trigger = super()._trigger_index(data)
        if type(trigger) != np.ndarray:
            self.N_new_samples += self.presamples - trigger # this amount of data will not be added in _add_data_to_buffer()
            self.N_acquired_samples_since_trigger += self.presamples - trigger
        return trigger
    
    def reset_buffer(self):
        """Resets 'finished' flag. 
        """
        self.rows_left = self.rows
        self.finished = False

class BaseAcquisition:
    """Parent acquisition class that should be used when creating new child acquisition source class.
    Child class should override methods the following methods:
    
    - self.__init__()
    
    - self.set_data_source()
    
    - self.terminate_data_source()
    
    - self.read_data()
    
    - self.clear_buffer() (optional)
    
    - self.get_sample_rate() (optional)
    
    For further information on how to override these methods, see the listed methods docstrings.
    
    Additionally, the __init__() or set_data_source() methods should override or be able to set the following attributes:
    
    - self._channel_names_init - list of original data channels names from source 
    
    - self._channel_names_video_init - list of original video channels names from source
    
    - self._channel_shapes_video_init - list of original video channels shapes from source
    
    - self.sample_rate = 0 - sample rate of acquisition source
    """
    all_acquisitions_ready = False # class property to indicate if all acquisitions are ready to start (not just this one)
    
    def __init__(self) -> None:
        """
        EDIT in child class. 
        
        Requirements:
        
        - Make sure to call super().__init__() AT THE BEGGINING of __init__() method.
        
        - Make sure to call self.set_trigger(1e20, 0, duration=1.0) AT THE END (used just for inititialization of buffer).
        """
        self.buffer_dtype = np.float64 # default dtype of data in ring buffer
        self.acquisition_name  = "DefaultAcquisition"
        
        # ----------------------------------------------------------------------------------------------
        # child class should override these attributes:
        self._channel_names_init        = [] # list of original channels names from source 
        self._channel_names_video_init  = [] # list of original channels names from source
        self._channel_shapes_video_init = [] # list of original video channels shapes from source
        self.sample_rate = 0
        # ----------------------------------------------------------------------------------------------	
        
        # these channel names variables are used to store channel names in the order they are added to the ring buffer
        # these variables are automatically modified by the BaseAcquisition class:
        self.channel_names_all = []   # list of all channel names 
        self.channel_names= []        # list of channel names with shape (1, )
        self.channel_names_video = [] # list of channel names with shape (M, N)
        self.channel_pos    = []      # list of tuples with start and end index positions of the data in the flattened ring buffer corresponding to each channel
        self.channel_shapes = []      # list of tuples with shapes of each channel (self.channel_names_all)
        
        self.virtual_channel_dict = {} # dictionary of virtual channels where key is virtual channel name and values are
                                       # tuples (function_to_use, list of indices from self.channel_names_all)
        
        # some flags required for proper operation of the class:
        self.is_running = True    # is acquisition running
        self.is_standalone = True # if this is part of bigger system or used as standalone object
        self.is_ready = False     # if acquisition is ready to start the acquisition
        self.continuous_mode = False # if acquisition is in continuous mode
        
        self.lock_acquisition = threading.Lock() # ensures acquisition class runs properly if used in multiple threads.
        
        self.N_samples_to_acquire = None # number of samples to acquire
        self.n_channels  = 0 # number of channels
        self.n_channels_trigger = 0
        
    def __repr__(self):
        """Returns string representation of the object.
        """
        def add_to_string(string_name, variable, string, padding):
            spaces = ' ' * (padding - len(string_name))
            string += f"{string_name}:{spaces} {variable}\n"
            return string

        string = ""
        padding = 20
        string = add_to_string("Acquisition name", self.acquisition_name, string, padding)
        string = add_to_string("Number of channels", self.n_channels, string, padding)
        string = add_to_string("Data channels", self.channel_names, string, padding)
        string = add_to_string("Video channels", self.channel_names_video, string, padding)
        string = add_to_string("Sample rate", f"{self.sample_rate} Hz", string, padding)
        string = add_to_string("Continuous mode", self.continuous_mode, string, padding)
        string = add_to_string("Standalone", self.is_standalone, string, padding)

        return string
    
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
        self._set_all_channels()
    
    def _set_all_channels(self) -> None:
        """
        Sets actual and virtual channels. This method is called at the end of set_data_source() method.
        """
        # 0) reset all channels:
        self.channel_names_all = []
        self.channel_shapes    = []
        self.channel_names     = []
        self.channel_names_video = []
        self.channel_pos = []
        
        # 1) from channels recieved from acquisition source, get their names and shapes:
        # For data channels:
        self.channel_names.extend(self._channel_names_init)
        self.channel_names_all.extend(self._channel_names_init)
        for channel_name in self._channel_names_init:
            self.channel_shapes.append( (1, ) )
        
        # For video channels:
        self.channel_names_video.extend(self._channel_names_video_init)
        self.channel_names_all.extend(self._channel_names_video_init)
        for video_channel_shape in self._channel_shapes_video_init:
            self.channel_shapes.append( video_channel_shape )
        
        # NOTE: never change that virtual channels are added after data and video channels!!!
        # 2) add virtual channels:
        for virt_ch_name in self.virtual_channel_dict.keys():
            func, source_channel_indices, first_arg_is_ref, args, kwargs = self.virtual_channel_dict[virt_ch_name ]
            
            shapes_used = [self.channel_shapes[ idx ] for idx in source_channel_indices] # here this should support multiple channels
            
            # test if function returns proper output:
            dummy_arrays = [np.random.rand(3, *shape ) for  shape in shapes_used   ] # test arrays, 1st dim 3 just for testing
            func_input = dummy_arrays + list(args)
            if first_arg_is_ref:
                output = func(self, *func_input, **kwargs) # this is just a test to get output shape
            else:
                output = func(*func_input, **kwargs)
                
            if type(output) != np.ndarray:
                raise ValueError('Virtual channel function must return numpy array of arbitrary shape and not int, float, tuple...')
            
            # data channel  -> output.shape = (3, 1)
            # video channel -> output.shape = (3, M, K)
            self.channel_names_all.append(virt_ch_name )
            if len(output.shape[1:]) == 1 and output.shape[1] == 1: # signal channel
                shape = (1, )
                self.channel_shapes.append( shape )
                self.channel_names.append(virt_ch_name )
            elif len(output.shape[1:]) == 2: # video channel
                shape = output.shape[1:]
                self.channel_shapes.append( shape )
                self.channel_names_video.append(virt_ch_name )
            else:
                shape = output.shape
                raise ValueError(f'Output shape {shape} of virtual channel {virt_ch_name} is not supported.\n'
                                 'Virtual channel function must return numpy array of shape (n_samples, M) and NOT (n_samples, ) or (n_samples, M, K) ...'
                                 )
        
        # 3) recalculate total number of channels: 
        self.n_channels = len(self.channel_names_all)
        self.n_channels_trigger  = 0
        self.channel_pos = []
        pos = 0
        for shape in self.channel_shapes:
            self.n_channels_trigger += np.prod(shape)
            pos_next = pos+np.prod(shape)
            self.channel_pos.append( (pos, pos_next) )
            pos = pos_next
    
    def add_virtual_channel(self, virtual_channel_name:str, source_channels:int|str|list, function:callable, *args, **kwargs)->None:
        """
        Add a virtual channel to the acquisition class.
        
        Args:
            virtual_channel_name (str): Name of the channel that will be created
            source_channel (int, str, list): list of name strings or indices of the channels in self.channel_names_all on which function will be applied.
                                        optionally, if only one channel is used, it can be given as a string or index form self.channel_names_all
            function (function): Function used on the channels. Takes channels' arrays as input and has to return a numpy array
                                 where the first dimension is the number of samples. If returned array has 2 dimensions, it is treated
                                 as data source, if it has 3 dimensions, it is treated as video source.
                                 The first argument of the function can be a reference to the acquisition class itself. This is useful
                                 if the function needs to access some of the acquisition class' attributes, for example data history.
                                 
            *args: additional arguments to be passed to the function (function passed as input argument to this method)
            **kwargs: additional keyword arguments to be passed to the function (function passed as input argument to this method)y

        Example 1:
            >>> def func(ch1, ch2): # ch1 and ch2 are numpy arrays
            >>>     # ch1 and ch2 are of shape (n_samples, 1) and NOT (1, )
            >>>     return ch1 + ch2 # returns numpy array of shape (n_samples, 1) or (n_samples, M, K)
            >>> acq.add_virtual_channel('ch1+ch2', ['ch1', 'ch2'], func)
        
        Example 2:
            >>> def virtual_channel_func(self, ch1):
            >>>     try:
            >>>         # At virtual channel definition, retrieving data or channel indices is not yet possible for all channels.
            >>>         # Therefore, we use a try-except block to properly execute channel definition.
            >>>         time, data = self.get_data(N_points=2)
            >>>         i_ch3 = self.get_channel_index("ch3", channel_type='data') # retrieve data index channel of ch1
            >>>         ch3_prev = data[-1, i_ch3] # retrieve last value of ch1
            >>>         
            >>>     except:
            >>>         # at channel definition, no data is available yet. Therefore, we set the previous value to 0.
            >>>         ch3_prev = 0
            >>>         
            >>>     # cumulative sum of ch1:
            >>>     ch1_cumsum = np.cumsum(ch1) + ch3_prev	
            >>>     return ch1_cumsum.reshape(-1,1) # reshape to (n_samples, 1)
            >>>
            >>> acq.add_virtual_channel('ch3', 'ch1', virtual_channel_func)
            
            
        """
        first_arg_is_ref = False
        
        if type(source_channels) == int or type(source_channels) == str:
            source_channels = [source_channels]
            
        self.terminate_data_source()
        # list comprehension to get indices of the channels:
        source_channels = [self.channel_names_all.index(ch) if type(ch) == str else ch for ch in source_channels]
        # check if first element is refrence to acquisition class:
        sig = inspect.signature(function)
        input_arguments = [param.name for param in sig.parameters.values()]
        if input_arguments[0] == 'self':
            first_arg_is_ref = True
        
        if 'self' in input_arguments[1:]:
            raise ValueError('Virtual channel function cannot can have only one reference to acquisition class as FIRST argument.')
        
        self.virtual_channel_dict[virtual_channel_name] = (function, source_channels, first_arg_is_ref, args, kwargs)

        
        self.set_data_source()
        self.terminate_data_source()
        # TODO: check if only self._set_all_channels() is enough - it is not, because some sources set 
        #       self.channel_names_video_init and self.channel_shapes_video_init in set_data_source() method
        
    
    def terminate_data_source(self)->None:
        """EDIT in child class.
        
        Properly closes/disconnects acquisition source after the measurement. The method should be able to handle mutliple calls in a row.
        """
        pass
            
    def read_data(self)->np.ndarray:
        """EDIT in child class.
        
        This method only reads data from the source and transforms data into standard format used by other methods.
        It is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must ALWAYS return a 2D numpy array of shape (n_samples, n_columns).
        
        
        **IMPORTANT**: 
        If some of the channels are videos (2D array - so shape is (n_samples, n_pixels_width, n_pixels_height)), 
        then data has to be reshaped to shape (n_samples, n_pixels_width*n_pixels_height). Then data from multiple
        sources have to be concatenated into one array of shape (n_samples, n_cols), where cols is the combined number 
        of pixel of all video sources and number of channels in data sources.
        
        For an example where data source has 2 video sources with resolution 300x200 and 2 data channels, the final 
        shape returned by this methods should be (n_samples, 300*200*2+2).
    
        For video sources, the shape of the video is automatically stored in self.channel_shapes_video_init when
        self.set_data_source() is called. When data is retrieved from the source, it is reshaped to (n_samples, n_pixels_width, n_pixels_height).
                  
        Returns:
            data (np.ndarray): 2D numpy array of shape (n_samples, n_columns)
        """
        pass
    
    def _read_all_channels(self)->np.ndarray:
        """
        Uses acquired data and process them to create data for virtual channels.
        This method is continuously called by self.acquire() method.
        
        Returns:
            data (np.ndarray): 2D numpy array of shape (n_samples, n_columns)
        """
        # read data from source:
        data = self.read_data() # shape (n_samples, n_cols) - flattened video channels (if video)
        
        # calculate data of virtual channels:
        if len(self.virtual_channel_dict.keys()) > 0:
            for virt_ch_name, (func, source_channel_indices, first_arg_is_ref, args, kwargs) in self.virtual_channel_dict.items():
                data_source_list = [
                    data[:, self.channel_pos[idx][0] : self.channel_pos[idx][1] ].reshape(-1, *self.channel_shapes[idx])
                    for idx in source_channel_indices
                ]
                func_input = data_source_list + list(args)
                if first_arg_is_ref:
                    data_virt_ch = func(self, *func_input, **kwargs)
                else:
                    data_virt_ch = func(*func_input, **kwargs)
                    
                if len(data_virt_ch.shape) == 1:
                    data_virt_ch = data_virt_ch.reshape(-1, 1)
                elif len(data_virt_ch.shape) == 2:
                    pass
                elif len(data_virt_ch.shape) == 3: # video channel!
                    data_virt_ch = data_virt_ch.reshape(-1, np.prod(data_virt_ch.shape[1:])) # flatten virtual channels
                else:
                    raise ValueError('Virtual channel function must return numpy array with shape:\n' \
                                    '(n_samples, 1) - signal channel OR (n_samples, n_pixels_width*n_pixels_height) - video channel')
                    
                data = np.concatenate((data, data_virt_ch), axis=1)

        return data
    
    def get_sample_rate(self)->float:
        """EDIT in child class (Optional).
        
        Returns sample rate of acquisition class.
        """
        return self.sample_rate
    
    def get_channel_index(self, channel_name:str, channel_type:str='data')->int:
        """Returns the index of the channel from either self.channel_names_all, self.channel_names or self.channel_names_video.
        The channel_type argument is used to specify which list to use. If index is used for retrieving channel data from array
        returned by self.get_data() then channel_type should depend on which type of data you are recieving.

        Args:
            channel_name (str): name of the channel
            channel_type (str): type of the channel. Can be 'all', 'data' or 'video'. Default is 'data'.
        Returns:
            channel_index (int): index of the channel
        """
        if channel_type == 'all':
            return self.channel_names_all.index(channel_name)
        elif channel_type == "data":
            return self.channel_names.index(channel_name)
        elif channel_type == "video":
            return self.channel_names_video.index(channel_name)
        else:
            raise ValueError("channel_type must be 'all', 'data' or 'video'.")
    
    def clear_buffer(self)->None:
        """EDIT in child class (Optional).
        
        The source buffer should be cleared with this method. It can either clear the buffer, or
        just read the data with self.read_data() and does not add/save data anywhere. By default, this method
        will read the data from the source and not add/save data anywhere.
        
        Returns None.
        """
        self.read_data()
            
    def stop(self)->None:
        """Stops acquisition run.
        """
        self.is_running = False
        
        # wait for the thread to finish if it exists:
        if hasattr(self, "background_thread") and threading.current_thread() == threading.main_thread():
            if self.background_thread.is_alive():
                self.background_thread.join()
            
    
    def acquire(self):
        """Acquires data from acquisition source and also properly saves the data to pyTrigger ringbuffer.
        Additionally it also stops the measurement run and terminates acquisition source properly.
        This method is continuously called in the run_acquisition() method.
        """
        with self.lock_acquisition: # lock to secure variables
            acquired_data = self._read_all_channels()
            self.Trigger.add_data(acquired_data)
            
        if self.Trigger.finished or not self.is_running:   
            self.stop()
            self.terminate_data_source()
        
    def run_acquisition(self, run_time:float=None, run_in_background:bool=False)->None:
        """
        Runs acquisition. This method is used to start the acquisition.
        
        Args:
            run_time (float): number of seconds for which the acquisition will run.
            run_in_background (bool): if True, acquisition will run in a separate thread.
        
        Returns:
            None
        """
        BaseAcquisition.all_acquisitions_ready = False # set all acquisitions to not ready. 
                                                       # NOTE: this is a class variable, it could mess semething up in the 
                                                       # future complex applications where sources are not started at once!
        self.is_ready   = False
        self.is_running = True
        
        if run_time is None:
            self._set_trigger_instance() # Again set the trigger instance, because it may have been changed
        else:
            self.update_trigger_parameters(duration=run_time, duration_unit='seconds')
            
        self.set_data_source() # start data source
        
        # if acquisition is used in some other classes, wait until all acquisition sources are ready:
        if not self.is_standalone:
            self.is_ready = True    # this source is ready (other may not be)
            while not BaseAcquisition.all_acquisitions_ready: # until every source is ready
                # NOTE: BaseAcquisition.all_acquisitions_ready is set to True by Core() class that handles multiple sources
                time.sleep(0.01)
                self.clear_buffer() # reads data, does not store in anywhere
                
                if not self.is_running: # in case the acquisition is stopped before it starts
                    break
                
            time.sleep(0.01)
            self.clear_buffer() # ensure buffer is cleared at least once. 
        else:
            # acquisition is being run as a standalone process, so no need to wait for other sources
            pass
        
        def _loop(): # main acquisition loop:
            if run_time == None:
                while self.is_running:
                    time.sleep(0.01)
                    self.acquire()
            else:
                N_total_samples = int(run_time*self.sample_rate)
                while self.is_running:  
                    if self.Trigger.N_acquired_samples >= N_total_samples:
                        self.is_running = False
                        
                    self.acquire()
                    time.sleep(0.01)
                    
        if run_in_background:
            self.background_thread = threading.Thread(target=_loop)
            self.background_thread.start() 
        else:
            _loop()        
       
    def set_continuous_mode(self, boolean:bool=True, measurement_duration:float=None)->None:
        """Sets continuous mode of the acquisition. 
        
        If True, acquisition will run indefinitely until externally stopped. If False, acquisition will run for a specified time.

        Args:
            boolean (bool, optional): Defaults to True.
            measurement_duration (float, optional): If not None, sets the duration of the measurement in seconds. It does NOT
                                                    update ring buffer size. Defaults to None. Based on measurement_duration, the number of total samples to be acquired is calculated. In this case the 
                                                    ring buffer size can be different to the number of samples to be acquired. If None, measurement duration measurement
                                                    will not stop until externally stopped. This means that after the ring buffer is filled, the oldest data will be
                                                    overwritten by the newest data. 
            
        Returns:
            None
            
        Examples:
            >>> # Setting continuous mode to True, setting buffer size to 10 seconds of data, measurement will run indefinitely:
            >>> acq.set_trigger(level=0, channel=0, duration=10.) # this will trigger measurement right away, buffer size will be 10 seconds
            >>> acq.set_continuous_mode(True, measurement_duration=None) # acquisition will run indefinitely until externally stopped
            
            >>> # Setting continuous mode to True, setting buffer size to 5 seconds of data, measurment will run for 10 seconds:
            >>> acq.set_trigger(level=0, channel=0, duration=5.) # this will trigger measurement right away, buffer size will be 5 seconds
            >>> acq.set_continuous_mode(True, measurement_duration=10.) # acquisition will run for 10 seconds, but buffer will store only 5 seconds of data
        """
        if boolean:
            self.continuous_mode = True
        else:
            self.continuous_mode = False
            
        if measurement_duration is not None:
            self.N_samples_to_acquire = int(measurement_duration*self.sample_rate)
        else:
            self.N_samples_to_acquire = None
                   
    def _set_trigger_instance(self)->None:
        """
        Creates CustomPyTrigger instance and sets its parameters.
        
        Args:
            None
            
        Returns:
            None
        """
        
        # convert source channel index to ringbuffer channel index:
        if len(self.channel_names) == 0: # this source has no data channels
            buffer_channel = 0 # set arbitrary channel 
            level = 1e20 # set a level that will not be triggered            
            # TODO: currently this is a little hacky, but it works. In the future, this should be changed
        else:  
            channel = self.trigger_settings['channel']
            # convert to index from self.channel_names_all:
            try:
                if type(channel) == str:
                    channel = self.channel_names_all.index(channel) if type(channel)==str else channel
                elif type(channel) == int:
                    channel = self.channel_names_all.index(self.channel_names[channel]) 
                else:
                    raise ValueError("Channel must be either string or integer")
            except:
                raise IndexError("Channel name not found in the list of available channels (self.channel_names)")
            
            buffer_channel = self.channel_pos[channel][0] # 1st index is the data channel position in the ring buffer
            level = self.trigger_settings['level']
        
        # Create trigger instance:
        self.Trigger = CustomPyTrigger( #pyTrigger
            rows=self.trigger_settings['duration_samples'], 
            channels=self.n_channels_trigger,
            trigger_type=self.trigger_settings['type'],
            trigger_channel=buffer_channel, 
            trigger_level=level,
            presamples=self.trigger_settings['presamples'],
            dtype=self.buffer_dtype)
        
        self.Trigger.continuous_mode = self.continuous_mode
        if self.continuous_mode:
            self.Trigger.N_samples_to_acquire = self.N_samples_to_acquire   
            
        #self.N_samples_to_acquire = self.trigger_settings["duration_samples"]      
        
    def set_trigger(self, level:float, channel:str|int, duration:float|int=1, duration_unit:str='seconds', presamples:int=0, type:str='abs')->None:
        """Set parameters for triggering the measurement. 
        
        Only one trigger channel is supported at the moment. Additionally trigger can only be set
        on 'data' channels. If trigger is needed on 'video' channels, a 'data' virtual channel has to be created
        using 'add_virtual_channel()' method, and then trigger can be set on this virtual channel.
        
        Args:
            level (float): trigger level
            channel (int, str): trigger channel (int or str). If str, it must be one of the channel names. If int, 
                                index from self.channel_names ('data' channels) has to be provided (NOTE: see the difference between
                                self.channel_names and self.channel_names_all).
            duration (float, int, optional): duration of the acquisition after trigger (in seconds or samples). Defaults to 1.
            duration_unit (str, optional): 'seconds' or 'samples'. Defaults to 'seconds'.
            presamples (int, optional): number of presamples to save. Defaults to 0.
            type (str, optional): trigger type: up, down or abs. Defaults to 'abs'.
            
        Returns:
            None
        """

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
        
        self._set_trigger_instance()
        
    def update_trigger_parameters(self, **kwargs)->None:
        """
        Updates trigger settings. See 'set_trigger()' method for possible parameters.
        """  
        for setting, value in kwargs.items():
            self.trigger_settings[setting] = value
            
        if self.trigger_settings['duration_unit'] == 'seconds':
            self.trigger_settings['duration_samples'] = int(self.sample_rate*self.trigger_settings['duration'])
            self.trigger_settings['duration_seconds'] = self.trigger_settings['duration']
         
        elif self.trigger_settings['duration_unit'] == 'samples':
            self.trigger_settings['duration_seconds'] = self.trigger_settings['duration']/self.sample_rate
            self.trigger_settings['duration_samples'] = self.trigger_settings['duration']
        
        self._set_trigger_instance()
        
    def activate_trigger(self, all_sources:bool=True)->None:
        """
        Sets trigger off. Useful if the acquisition class is trigered by another process.
        This trigger can also trigger other acquisition sources by setting property class
        """
        if all_sources:
            CustomPyTrigger.triggered_global = True
        else:
            self.Trigger.triggered = True

    def reset_trigger(self, all_sources:bool=True)->None:
        """
        Resets trigger.
        
        Parameters:
            all_sources (bool): if True, resets trigger of all acquisition sources. If False, resets only this acquisition source.
                                Currently, this parameter is not used, it is always set to True.
        """
        # TODO: check if all_sources parameter can be removed.
        CustomPyTrigger.triggered_global = False
        self.Trigger.triggered = False
        
    def is_triggered(self)->bool:
        """
        Checks if acquisition class has been triggered during measurement.

        Returns:
            bool: True/False if triggered
        """
        return self.Trigger.triggered
        
    def _all_acquisitions_ready(self)->None:
        """Sets ALL acquisition sources (not only this one) to ready state. Should not be generally used.
        This method is normally used  by Core() class to set all acquisition sources to ready state.
        """
        BaseAcquisition.all_acquisitions_ready = True
    
    def _reshape_data(self, flattened_data:np.ndarray, data_to_return:str)->np.ndarray|list:
        """Reshapes channel arrays to the original shape.

        Args:
            flattened_data (np.ndarray): flattened array from ring buffer
            data_to_return (str): 'video' or 'data'. If 'data', only data channels are returned as an array
                                   with shape (n_samples, n_data_channels). If 'video', only video channels are returned
                                   as a list of 3D arrays with shape (n_samples, height, width). array positions in the list
                                   correspond to the order of video channels in self.channel_names_video.
                                   if 'flattened', returns flattened data array with shape (n_samples, n_ringbuffer_channels).

        Returns:
            np.ndarray, list: 
            
                - if 'data' is requested, returns array with shape (n_samples, n_data_channels).
                
                - if 'video' is requested, returns list of 3D arrays with shape (n_samples, height, width).
                
                - if 'flattened' is requested, returns flattened data array with shape (n_samples, n_ringbuffer_channels).
        """
        if data_to_return=="video":
            channels = [self.channel_names_all.index(name) for name in self.channel_names_video]
            if len(channels)==0:
                raise ValueError(f"No video channels are defined in {self.acquisition_name}.")
            
            data_return = []
            for channel in channels:
                shape = self.channel_shapes[channel]
                pos = self.channel_pos[channel]
                data_return.append( flattened_data[:, pos[0]:pos[1]].reshape( (flattened_data.shape[0], *shape) ) )
                
        elif data_to_return=="data":
            channels = [self.channel_names_all.index(name) for name in self.channel_names]
            if len(channels)==0:
                raise ValueError(f"No data channels are defined in {self.acquisition_name}.")
            
            pos_list = [np.arange(self.channel_pos[channel][0], self.channel_pos[channel][1]) for channel in channels]
            pos_list = np.concatenate(pos_list)
            
            data_return = flattened_data[:, pos_list]
        elif data_to_return=="flattened": # return flattened buffer
            data_return = flattened_data
        else:
            raise ValueError(f"Unknown data_to_return parameter: {data_to_return}. Possible values are 'video', 'data' or 'flattened'")
        
        return data_return
            
    def get_data(self, N_points:int|str|None=None, data_to_return:str="data")->tuple:
        """Reads and returns data from the buffer.
        
        Args:
            N_points (int, str, None): number of last N points to read from pyTrigger buffer.
                if N_points="new", then only new points will be retrieved.
                if None all samples are returned.
            data_to_return (str): 'video', 'data' or 'flattened'. If 'data', only data channels are returned as an array
    
        Returns:
            tuple: (time, data) - np.ndarray 1D time vector and measured data. Data shape depends on 'data_to_return' parameter:
            
                - if 'data' is requested, returns array with shape (n_samples, n_data_channels).
                
                - if 'video' is requested, returns list of 3D arrays with shape (n_samples, height, width).
                
                - if 'flattened' is requested, returns flattened data array with shape (n_samples, n_ringbuffer_channels).
                            
        IMPORTANT: if N_points = "new" is used for retrieving new results during measurement and Core() class is used for measurement control,
        then periodic saving should be turned OFF in Core() class. In other words, Core.run() method should be called with save_interval=None.
        """        
        if N_points is None:
            data = self.Trigger.get_data()[-self.Trigger.N_acquired_samples_since_trigger:]
            
        elif N_points == "new":
            with self.lock_acquisition: # lock acquisition to avoid reading data while it is being written
                data = self.Trigger.get_data_new()
        else:
            data = self.Trigger.get_data()[-N_points:]
                
        N = self.Trigger.N_acquired_samples_since_trigger
        time = np.arange(N-data.shape[0], N)/self.sample_rate     
        data_return = self._reshape_data(data, data_to_return)
    
        return time, data_return
    
    def get_data_PLOT(self, data_to_return:str="data")->np.ndarray|list:
        """Reads only new data from pyTrigger ring buffer and returns it.
        
        This method is used only for plotting purposes and SHOULD NOT BE USED for any other purpose. Additionally,
        it does not return time vector, only data. See get_data() method for more details. Data returned in this
        method is the same as in get_data().
        
        Returns:
            np.ndarray, list:
            
                    - if 'data' is requested, returns array with shape (n_samples, n_data_channels).
                    
                    - if 'video' is requested, returns list of 3D arrays with shape (n_samples, height, width).
                    
                    - if 'flattened' is requested, returns flattened data array with shape (n_samples, n_ringbuffer_channels)
        """
        with self.lock_acquisition: # lock acquisition to avoid reading data while it is being written
            data = self.Trigger.get_data_new_PLOT()
        
        data_return = self._reshape_data(data, data_to_return)
        
        return data_return
    
    def get_measurement_dict(self, N_points:int|str=None)->dict:
        """Reads data from pyTrigger ring buffer using self.get_data() method and returns a dictionary containing all relevant information about the measurement. 

        Args:
            N_points (None, int, str): Number of points to get from pyTrigger ringbuffer. If type(N_points)==int then N_points
                                       last samples are returned. If N_points=='new', only new points after trigger event are returned.
                                       If None, all samples are returned. Defaults to None.

        Returns:
            dict: keys and values are the following:
                
                - 'time': 1D array
                 
                - 'channel_names': list of channel names
                
                - 'data': 2D array (n_samples, n_data_channels)
                
                - 'channel_names_video': list of video channel names
                
                - 'video': list of 3D arrays (n_samples, height, width)
                
                - 'sample_rate': sample rate of acquisition
                   
        IMPORTANT: 
        If N_points = "new" is used for retrieving new results during measurement and Core() class is used for measurement control,
        then periodic saving should be turned OFF in Core() class. In other words, Core.run() method should be called with save_interval=None.
        """
        measurement_dict = {}
        
        time, data = self.get_data(N_points=N_points, data_to_return="flattened")
        
        if len(self.channel_names_video) > 0:
            # get data only:
            idx_signal_channels = [self.channel_names_all.index(name) for name in self.channel_names]
            pos_list = [np.arange(self.channel_pos[channel][0], self.channel_pos[channel][1]) for channel in idx_signal_channels]
            if len(pos_list) > 0:
                pos_list = np.concatenate(pos_list)
                data_only = data[:, pos_list]
            else:
                data_only = np.array([]) # no data channels
                
            # get video only:
            idx_video_channels = [self.channel_names_all.index(name) for name in self.channel_names_video]
            video_only = []
            for channel in idx_video_channels:
                shape = self.channel_shapes[channel]
                pos = self.channel_pos[channel]
                video_only.append( data[:, pos[0]:pos[1]].reshape( (data.shape[0], *shape) ) )
            
            # save video and data separately
            measurement_dict['time'] = time
            
            measurement_dict['channel_names'] = self.channel_names.copy()
            measurement_dict['data']  = data_only
            
            measurement_dict['channel_names_video'] = self.channel_names_video.copy()
            measurement_dict['video'] = video_only
            
        else: # no video, flattened array is actually only data:
            measurement_dict['time'] = time
            measurement_dict['channel_names'] = self.channel_names.copy()
            measurement_dict['data'] = data
        
        if hasattr(self, 'sample_rate'):
            measurement_dict['sample_rate'] = self.sample_rate
        else:
            measurement_dict['sample_rate'] = None
            
        return measurement_dict
    
    def save(self, name:str, root:str='', timestamp:bool=True, comment:str=None)->None:
        """Save acquired data.
        
        Args:
            name (str): filename
            root (str, optional): directory to save to. Defaults to ''.
            timestamp (bool, optional): include timestamp before 'filename'. Defaults to True.
            comment (str, optional): commentary on the saved file. Defaults to None.
            
        Returns:
            None
        """
        measurement_dict = self.get_measurement_dict()
        
        if comment is not None:
            measurement_dict['comment'] = comment
        
        if not os.path.exists(root):
            os.mkdir(root)

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        filename = f'{stamp}{name}.pkl'
        path = os.path.join(root, filename)
        pickle.dump(measurement_dict, open(path, 'wb'), protocol=-1)

