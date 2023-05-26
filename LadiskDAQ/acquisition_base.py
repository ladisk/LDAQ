import os
import numpy as np
import pickle
import datetime
import time
import threading

from pyTrigger import pyTrigger, RingBuffer2D
from ctypes import *


class CustomPyTrigger(pyTrigger):
    """
    Upgrades pyTrigger class with features needed for acquisition class BaseAcquisition.
    
    :param rows: # of rows
    :param channels: # of channels
    :param trigger_channel: the channel used for triggering
    :param trigger_level: the level to cross, to start trigger
    :param trigger_type: 'up' is default, possible also 'down'/'abs'
    :param presamples: # of presamples
    """
    triggered_global = False
    #def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
    def __init__(self, rows=5120, channels=4, trigger_channel=0,
                 trigger_level=1., trigger_type='up', presamples=1000,
                 dtype=np.float64):
        
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

    def _add_data_to_buffer(self, data):
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
        
    def _add_data_chunk(self, data):
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
            self.N_triggers        += 1
            self.first_trigger      = False
        return 
    
    def get_data_new(self):
        """Retrieves any new data from ring buffer after trigger event that has been not yet retrieved.

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
        
    def get_data_new_PLOT(self):
        """Retrieves any new data from ring buffer that has been not yet retrieved. 
           This is used for plotting purposes only.

        Returns:
            np.ndarray: data of shape (rows, channels)
        """
        if self.N_new_samples_PLOT > 0:
            data = self.ringbuff.get_data()[-self.N_new_samples_PLOT:]
            self.N_new_samples_PLOT = 0
            return data
        else:
            return np.empty(shape=(0, self.channels))
    
    def _trigger_index(self, data):
        """Upgrades parent _trigger_index() method. Beside searching for trigger event, it
           adds amount of samples missed by _add_data_to_buffer() in case of use of presamples.
        """
        trigger = super()._trigger_index(data)
        if type(trigger) != np.ndarray:
            self.N_new_samples += self.presamples - trigger # this amount of data will not be added in _add_data_to_buffer()
            self.N_acquired_samples_since_trigger += self.presamples - trigger
        return trigger
    
    def reset_buffer(self):
        self.rows_left = self.rows
        self.finished = False

class BaseAcquisition:
    """Parent acquisition class that should be used when creating new child acquisition source class.
    Child class should override methods the following methods:
    - self.read_data()
    - self.terminate_data_source()
    - self.set_data_source()
    - self.clear_buffer()
    - self.get_sample_rate() (optional)
    
    Additionally, the __init__() method should override the following attributes:
    - self.n_channels 
    - self.n_channels_trigger (same as n_channels if source is not a camera)
    - self.channel_names 
    - self.sample_rate
    - self.acquisition_name (optional)
    
    NOTE: the __init__() method should call self.set_trigger(1e20, 0, duration=1.0)
    at the end of __init__ method to set trigger eventhough not used.

    Returns:
        _type_: _description_
    """
    all_acquisitions_ready = False # class property to indicate if all acquisitions are ready to start (not jsut this one)
    
    def __init__(self):
        """EDIT in child class"""
        self.buffer_dtype = np.float64 # default dtype of data in ring buffer
        self.acquisition_name = "DefaultAcquisition"
        self.channel_names = [] # list of channel names with shape (1, )
        self.channel_names_all = [] # list of all channel names 
        self.channel_names_video = [] # list of channel names with shape (M, N)
        self.channel_pos = [] # list of tuples with start and end index positions of the data in the flattened ring buffer corresponding to each channel
        self.channel_shapes = [] # list of tuples with shapes of each channel (self.channel_names_all)
        
        self.is_running = True
        self.is_standalone = True # if this is part of bigger system or used as standalone object
        self.is_ready = False
    
        self.lock_acquisition = threading.Lock() # ensures acquisition class runs properly if used in multiple threads.
        
        self.continuous_mode = False
        self.N_samples_to_acquire = None
        # child class needs to have variables below:
        self.n_channels  = 0
        self.n_channels_trigger = 0
        self.sample_rate = 0
        
    def read_data(self):
        """EDIT in child class
        
        This method only reads data from the source and transforms data into standard format used by other methods.
        This method is called within self.acquire() method which properly handles acquiring data and saves it into 
        pyTrigger ring buffer.
        
        Must return a 2D numpy array of shape (n_samples, n_columns).
        """
        pass

    def terminate_data_source(self):
        """EDIT in child class
        
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        pass

    def set_data_source(self):
        """EDIT in child class
        Properly sets acquisition source before measurement is started.
        Should be set up in a way that it is able to be called multiple times in a row without issues.    
        """
        pass
    
    def get_sample_rate(self):
        """EDIT in child class
        
        Returns sample rate of acquisition class.
        This function is also useful to compute sample_rate estimation if no sample rate is given
        
        Returns self.sample_rate
        """
        return self.sample_rate
    
    def clear_buffer(self):
        """EDIT in child class
        
        The source buffer should be cleared with this method. Either actually clears the buffer, or
        just reads the data with self.read_data() and does not add/save data anywhere.
        
        Returns None.
        """
        self.read_data()

    # The following methods should work without changing.
    def stop(self):
        """Stops acquisition run.
        """
        self.is_running = False
        
        # what for the thread to finish if it exists:
        if hasattr(self, "background_thread") and threading.current_thread() == threading.main_thread():
            if self.background_thread.is_alive():
                self.background_thread.join()
            
    
    def acquire(self):
        """Acquires data from acquisition source and also properly saves the data to pyTrigger ringbuffer.
        Additionally it also stops the measurement run and terminates acquisition source properly.
        """
        with self.lock_acquisition: # lock to secure variables
            acquired_data = self.read_data()
            self.Trigger.add_data(acquired_data)
            
        if self.Trigger.finished or not self.is_running:      
        #if not self.is_running:       
            self.stop()
            self.terminate_data_source()

    def get_data(self, N_points=None, data_to_return="data"):
        """Reads and returns data from the pyTrigger buffer.
        :param N_points (int, str, None): number of last N points to read from pyTrigger buffer. 
                            if N_points="new", then only new points will be retrieved.
                            if None all samples are returned.
        Returns:
            tuple: (time, data) - 1D time vector and 2D measured data, both np.ndarray
        """        
        if N_points is None:
            data = self.Trigger.get_data()[-self.Trigger.N_acquired_samples_since_trigger:]
            
        elif N_points == "new":
            with self.lock_acquisition:
                data = self.Trigger.get_data_new()
        else:
            data = self.Trigger.get_data()[-N_points:]
                
        time = np.arange(data.shape[0])/self.sample_rate     
        return time, data
    
    def get_data_PLOT(self, data_to_return="data"):
        """Reads only new data from pyTrigger ring buffer and returns it.
        NOTE: this method is used only for plotting purposes and should not be used for any other purpose.
              also it does not return time vector, only data.
        Returns:
            array: 2D numpy array of shape (N_new_samples, n_channels)
        """
        with self.lock_acquisition:
            return self.Trigger.get_data_new_PLOT()
    
    def get_measurement_dict(self, N_points=None):
        """Reads data from pyTrigger ring buffer using self.get_data() method and returns a dictionary
           {'data': data, 'time': time, 'channel_names': self.channel_names, 'sample_rate' : sample_rate}

        Args:
            N_points (None, int, str): Number fo points to get from pyTrigger ringbuffer. If type(N_points)==int then N_points
                                       last samples are returned. If N_points=='new', only new points after trigger event are returned.
                                       If None, all samples are returned. Defaults to None.

        Returns:
            dict: {'data': data, 'time': time, 'channel_names': self.channel_names, 'sample_rate' : sample_rate}
        """
        self.measurement_dict = {}
        
        time, data = self.get_data(N_points=N_points, data_to_return="flattened")
        
        if len(self.channel_names_video) > 0:
            # get data only:
            idx_data_channels = [self.channel_names_all.index(name) for name in self.channel_names]
            pos_list = [np.arange(self.channel_pos[channel][0], self.channel_pos[channel][1]) for channel in idx_data_channels]
            pos_list = np.concatenate(pos_list)
            data_only = data[:, pos_list]
            
            # get video only:
            idx_video_channels = [self.channel_names_all.index(name) for name in self.channel_names_video]
            video_only = []
            for channel in idx_video_channels:
                shape = self.channel_shapes[channel]
                pos = self.channel_pos[channel]
                video_only.append( data[:, pos[0]:pos[1]].reshape( (data.shape[0], *shape) ) )
            
            # save video and data separately
            self.measurement_dict['time'] = time
            
            self.measurement_dict['channel_names'] = self.channel_names
            self.measurement_dict['data']  = data_only
            
            self.measurement_dict['channel_names_video'] = self.channel_names_video
            self.measurement_dict['video'] = video_only
            
        else: # no video, flattened array is actually only data:
            self.measurement_dict['time'] = time
            self.measurement_dict['channel_names'] = self.channel_names
            self.measurement_dict['data'] = data
        
        if hasattr(self, 'sample_rate'):
            self.measurement_dict['sample_rate'] = self.sample_rate
        else:
            self.measurement_dict['sample_rate'] = None
            
        return self.measurement_dict
        
    def run_acquisition(self, run_time=None, run_in_background=False):
        """
        Runs acquisition.
        :params: run_time - (float) number of seconds for which the acquisition will run. 
            If None acquisition runs indefinitely until self.is_running variable is set
            False externally (i.e. in a different process)
        :params: run_in_background - (bool) if True, acquisition will run in a separate thread.
        """
        BaseAcquisition.all_acquisitions_ready = False 
        self.is_ready = False
        self.is_running = True
        
        if run_time is None:
            self._set_trigger_instance()
        else:
            self.update_trigger_parameters(duration=run_time, duration_unit='seconds')
            
        self.set_data_source()
        
        # if acquisition is used in some other classes, wait until all acquisition sources are ready:
        if not self.is_standalone:
            self.is_ready = True    # this source is ready (other may not be)
            while not BaseAcquisition.all_acquisitions_ready: # until every source is ready
                time.sleep(0.01)
                self.clear_buffer()                           # reads data, does not store in anywhere
                if not self.is_running:
                    break
                
            time.sleep(0.01)
            self.clear_buffer() # ensure buffer is cleared at least once. 
        else:
            # acquisition is being run as a standalone process, so no need to wait for other sources
            pass
        
        def _loop():
            # main acquisition loop:
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
       
    def set_continuous_mode(self, boolean=True, measurement_duration=None):
        """Sets continuous mode of the acquisition. If True, acquisition will run indefinitely until
           externally stopped. If False, acquisition will run for a specified time.

        Args:
            boolean (bool, optional): Defaults to True.
            measurement_duration (float, optional): If not None, sets the duration of the measurement in seconds.
            NOTE: Based on measurement duration, the number of total samples to be acquired is calculated. In this case the 
            ring buffer size can be different to the number of samples to be acquired. If None, measurement duration is 
            set to the size of the ring buffer.
        """
        if boolean:
            self.continuous_mode = True
        else:
            self.continuous_mode = False
            
        if measurement_duration is not None:
            self.N_samples_to_acquire = int(measurement_duration*self.sample_rate)
                   
    def _set_trigger_instance(self):
        """Creates PyTrigger instance.
        """
        self.Trigger = CustomPyTrigger( #pyTrigger
            rows=self.trigger_settings['duration_samples'], 
            channels=self.n_channels_trigger,
            trigger_type=self.trigger_settings['type'],
            trigger_channel=self.trigger_settings['channel'], 
            trigger_level=self.trigger_settings['level'],
            presamples=self.trigger_settings['presamples'],
            dtype=self.buffer_dtype)
        
        self.Trigger.continuous_mode = self.continuous_mode
        if self.continuous_mode:
            self.Trigger.N_samples_to_acquire = self.N_samples_to_acquire           
        
    def set_trigger(self, level, channel, duration=1, duration_unit='seconds', presamples=0, type='abs'):
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
        
        self._set_trigger_instance()
        
    def update_trigger_parameters(self, **kwargs):
        """
        Updates trigger settings. See 'set_trigger' method for possible parameters.
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
        
    def activate_trigger(self, all_sources=True):
        """Sets trigger off. Useful if the acquisition class is trigered by another process.
            This trigger can also trigger other acquisition sources by setting property class
        """
        if all_sources:
            CustomPyTrigger.triggered_global = True
        else:
            self.Trigger.triggered = True

    def reset_trigger(self):
        """Resets trigger.
        """
        CustomPyTrigger.triggered_global = False
        self.Trigger.triggered = False
        
    def is_triggered(self):
        """Checks if acquisition class has been triggered during measurement.

        Returns:
            bool: True/False if triggered
        """
        return self.Trigger.triggered
        
    def _all_acquisitions_ready(self):
        """Sets ALL acquisition sources (not only this one) to ready state. Should not be generally used.
        """
        BaseAcquisition.all_acquisitions_ready = True
    
    def save(self, name, root='', timestamp=True, comment=None):
        """Save acquired data.
        
        :param name: filename
        :param root: directory to save to
        :param timestamp: include timestamp before 'filename'
        :param comment: commentary on the saved file
        """
        self.measurement_dict = self.get_measurement_dict()
        
        if comment is not None:
            self.measurement_dict['comment'] = comment
        
        if not os.path.exists(root):
            os.mkdir(root)

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        filename = f'{stamp}{name}.pkl'
        path = os.path.join(root, filename)
        pickle.dump(self.measurement_dict, open(path, 'wb'), protocol=-1)

