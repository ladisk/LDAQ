import numpy as np
import time
import copy

from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.Task import Task
from nidaqmx._lib import lib_importer
from .daqtask import DAQTask

from ctypes import *

from .ni_task import NITask
from ..acquisition_base import BaseAcquisition


class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class."""

    def __init__(self, task_name, acquisition_name=None):
        """Initialize the task.

        :param task_name: Name of the task from NI Max
        """
        super().__init__()

        try:
            DAQmxClearTask(taskHandle_acquisition)
        except:
            pass

        try:
            lib_importer.windll.DAQmxClearTask(taskHandle_acquisition)
        except:
            pass
        
        self.task_terminated = True

        self.task_base = task_name
        if isinstance(task_name, str):
            self.NITask_used = False
        elif isinstance(task_name, NITask):
            self.NITask_used = True
        else:
            raise TypeError("task_name has to be a string or NITask object.")

        self.set_data_source()
        self.acquisition_name = self.task_name if acquisition_name is None else acquisition_name

        self.sample_rate = self.Task.sample_rate
        self.channel_names = self.Task.channel_list
        self.n_channels = self.Task.number_of_ch
        self.n_channels_trigger = self.n_channels

        if not self.NITask_used:
            glob_vars = globals()
            glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=600)

    def clear_task(self):
        """Clear a task."""
        self.Task.clear_task(wait_until_done=False)
        time.sleep(0.1)
        del self.Task

    def terminate_data_source(self):
        self.task_terminated = True
        self.clear_task()
        
    def read_data(self):
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def clear_buffer(self):
        self.Task.acquire_base()
    
    def set_data_source(self):
        if self.task_terminated:
            if self.NITask_used:
                channels_base = copy.deepcopy(self.task_base.channels)
                self.Task = NITask(self.task_base.task_name, self.task_base.sample_rate, self.task_base.settings_file)
                self.task_name = self.task_base.task_name

                for channel_name, channel in channels_base.items():
                    self.Task.add_channel(
                        channel_name, 
                        channel['device_ind'],
                        channel['channel_ind'],
                        channel['sensitivity'],
                        channel['sensitivity_units'],
                        channel['units'],
                        channel['serial_nr'],
                        channel['scale'],
                        channel['min_val'],
                        channel['max_val'])
            else:
                self.Task = DAQTask(self.task_base)
            
            self.task_terminated = False
        
        if self.NITask_used:
            if not hasattr(self.Task, 'task'):
                self.Task.initiate()

    def run_acquisition(self, run_time=None):        

        if self.NITask_used:
            BaseAcquisition.all_acquisitions_ready = False 
            self.is_ready = False
            self.is_running = True
            
            if run_time is None:
                self._set_trigger_instance()
            else:
                self.update_trigger_parameters(duration=run_time, duration_unit='seconds')
            
            self.set_data_source()
            glob_vars = globals()
            glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        super().run_acquisition(run_time)