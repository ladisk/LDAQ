import os
from collections import deque
import numpy as np
import pickle
import datetime
import time

from pyTrigger import pyTrigger

from .daqtask import DAQTask


class BaseAcquisition:
    def __init__(self):
        self.channel_names = []
        self.is_running = True
    
    def acquire(self):
        pass

    def run_acquisition(self):
        self.plot_data = []

        while self.is_running:
            self.acquire()


class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class."""

    def __init__(self, task_name):
        """Initialize the task.

        :param task_name: Name of the task from NI Max
        """
        super().__init__()

        self.task_name = task_name
        self.Task = DAQTask(self.task_name)
        self.channel_names = self.Task.channel_list
        self.sample_rate = self.Task.sample_rate

    def clear_task(self):
        """Clear a task."""
        self.Task.clear_task(wait_until_done=False)
        del self.Task

    def stop(self):
        self.is_running = False

    def acquire(self):
        self.Task.acquire()

        acquired_data = self.Task.data.T
        self.plot_data = np.vstack((self.plot_data, acquired_data))
        self.Trigger.add_data(acquired_data)

        if self.Trigger.finished or not self.is_running:
            self.data = self.Trigger.get_data()

            self.stop()
            self.clear_task()

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

    def run_acquisition(self):
        self.is_running = True
        
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)

        self.plot_data = np.zeros((1, len(self.channel_names)))
        self.set_trigger_instance()

        while self.is_running:
            self.acquire()

    def save(self, name, root='', save_columns='All', timestamp=True, comment=''):
        self.data_dict = {
            'data': self.data,
            'sample_rate': self.sample_rate,
            'channel_names': self.channel_names,
            'comment': comment,
        }

        if save_columns != 'All':
            self.data_dict['data'] = np.array(self.data_dict['data'])[:, save_columns]
            self.data_dict['channel_names'] = [_ for i, _ in enumerate(self.data_dict['channel_names']) if i in save_columns]

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        self.filename = f'{stamp}{name}.pkl'
        self.path = os.path.join(root, self.filename)
        pickle.dump(self.data_dict, open(self.path, 'wb'), protocol=-1)