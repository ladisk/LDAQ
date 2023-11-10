import numpy as np
import time
import copy

try:
    from PyDAQmx.DAQmxFunctions import *
    from PyDAQmx.Task import Task
    from nidaqmx._lib import lib_importer
    from .daqtask import DAQTask
except:
    pass

import typing

from ctypes import *

from .ni_task import NITask
from ..acquisition_base import BaseAcquisition

#TODO: remove pyDAQmx completely and use only nidaqmx
class NIAcquisition(BaseAcquisition):
    """National Instruments Acquisition class, compatible with any NI acquisition device that is supported by NI-DAQmx library.
    
    To use this class, you need to install NI-DAQmx library found on this link:
    https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html#494676
    
    Installation instructions:
    
    - Download NI-DAQmx from the link listed above.
    
    - Install NI-DAQmx.
    """

    def __init__(self, task_name: typing.Union[str, object], acquisition_name: typing.Optional[str] = None) -> None:
        """Initialize the acquisition task.

        Args:
            task_name (str, class object): Name of the task from NI Max or class object created with NITask() class using nidaqmx library.
            acquisition_name (str, optional): Name of the acquisition. Defaults to None, in which case the task name is used.
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
            self.task_name = task_name
        elif isinstance(task_name, NITask):
            self.NITask_used = True
            self.task_name = self.task_base.task_name
        else:
            raise TypeError("task_name has to be a string or NITask object.")

        self.set_data_source() # the data source must be set to red the number of channels and sample rate
        self.acquisition_name = self.task_name if acquisition_name is None else acquisition_name

        self.sample_rate = self.Task.sample_rate
        self._channel_names_init = self.Task.channel_list

        self.terminate_data_source() # clear the data source, will be set up later

        # if not self.NITask_used:
        #     glob_vars = globals()
        #     glob_vars['taskHandle_acquisition'] = self.Task.taskHandle

        # set default trigger, so the signal will not be trigered:
        self.set_trigger(1e20, 0, duration=1.0)

    def clear_task(self):
        """Clear a task."""
        if hasattr(self, "Task"):
            self.Task.clear_task(wait_until_done=False)
            time.sleep(0.1)
            del self.Task
        else:
            pass

    def terminate_data_source(self):
        """Properly closes the data source.
        """
        self.task_terminated = True
        self.clear_task()
        
    def read_data(self):
        """Reads data from device buffer and returns it.

        Returns:
            np.ndarray: numpy array with shape (n_samples, n_channels)
        """
        self.Task.acquire(wait_4_all_samples=False)
        return self.Task.data.T
    
    def clear_buffer(self):
        """
        Clears the buffer of the device.
        """
        self.Task.acquire_base()
    
    def set_data_source(self):
        """Sets the acquisition device to properly start the acquisition. This function is called before the acquisition is started.
           It is used to properly initialize the device and set the data source channels and virtual channels.
        """
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
                
        super().set_data_source()

    def run_acquisition(self, run_time=None, run_in_background=False):        
        """
        Runs acquisition. This is the method one should call to start the acquisition.

        Args:
            run_time (float): number of seconds for which the acquisition will run.
            run_in_background (bool): if True, acquisition will run in a separate thread.

        Returns:
            None
        """
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

        super().run_acquisition(run_time, run_in_background=run_in_background)