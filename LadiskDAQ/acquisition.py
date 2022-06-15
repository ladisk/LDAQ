import os
import numpy as np
import pickle
import datetime
import time

from pyTrigger import pyTrigger

from .daqtask import DAQTask


class BaseAcquisition:
    def __init__(self):
        """EDIT in child class"""
        self.channel_names = []
        self.plot_data = []
        self.is_running = True

    def read_data(self):
        """EDIT in child class
        
        This code acquires data. 
        
        Must return a 2D array of shape (n_samples, n_columns).
        """
        pass

    def clear_data_source(self):
        """EDIT in child class"""
        pass

    def set_data_source(self):
        """EDIT in child class"""
        pass

    # The following methods should work without changing.
    def stop(self):
        self.is_running = False
    
    def acquire(self):
        acquired_data = self.read_data()
        self.plot_data = np.vstack((self.plot_data, acquired_data))
        self.Trigger.add_data(acquired_data)
            
        if self.Trigger.finished or not self.is_running:
            self.data = self.Trigger.get_data()

            self.stop()
            self.clear_data_source()

    def run_acquisition(self):
        self.is_running = True

        self.set_data_source()

        self.plot_data = np.zeros((2, len(self.channel_names)))
        self.set_trigger_instance()

        while self.is_running:
            self.acquire()

    def set_trigger(self, level, channel, duration=1, duration_unit='seconds', presamples=100, type='abs'):
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

    def set_trigger_instance(self):
        self.Trigger = pyTrigger(
            rows=self.trigger_settings['duration_samples'], 
            channels=self.Task.number_of_ch,
            trigger_type=self.trigger_settings['type'],
            trigger_channel=self.trigger_settings['channel'], 
            trigger_level=self.trigger_settings['level'],
            presamples=self.trigger_settings['presamples'])
    
    def save(self, name, root='', save_channels='All', timestamp=True, comment=''):
        """Save acquired data.
        
        :param name: filename
        :param root: directory to save to
        :param save_channels: channel indices that are save. Defaults to 'All'.
        :param timestamp: include timestamp before 'filename'
        :param comment: commentary on the saved file
        """
        self.data_dict = {
            'data': self.data,
            'channel_names': self.channel_names,
            'comment': comment,
        }

        if hasattr(self, 'sample_rate'):
            self.data_dict['sample_rate'] = self.sample_rate

        if save_channels != 'All':
            self.data_dict['data'] = np.array(self.data_dict['data'])[:, save_channels]
            self.data_dict['channel_names'] = [_ for i, _ in enumerate(self.data_dict['channel_names']) if i in save_channels]

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        self.filename = f'{stamp}{name}.pkl'
        self.path = os.path.join(root, self.filename)
        pickle.dump(self.data_dict, open(self.path, 'wb'), protocol=-1)


class ADAcquisition(BaseAcquisition):
    def __init__(self, port_nr):
        super.__init__()


class SerialAcquisition(BaseAcquisition):
    def __init__(self, port_nr):
        super.__init__()


class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class."""

    def __init__(self, task_name):
        """Initialize the task.

        :param task_name: Name of the task from NI Max
        """
        super().__init__()

        self.task_name = task_name

        # TODO: clear task if it exists
        #try:
            #clear task if it exist 
            # done with  DAQmxClearTask(taskHandle) in Task.py
            # morš najdt handle pointer od taska in potem z zgornjo funkcijo clearat task
            # Klemen boš ti ann? :D
        #except:
            # pass
        
        self.Task = DAQTask(self.task_name)


        self.channel_names = self.Task.channel_list
        self.sample_rate = self.Task.sample_rate

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def clear_task(self):
        """Clear a task."""
        self.Task.clear_task(wait_until_done=False)
        del self.Task

    def clear_data_source(self):
        return self.clear_task()

    def read_data(self):
        self.Task.acquire()
        return self.Task.data.T
    
    def set_data_source(self):
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)

