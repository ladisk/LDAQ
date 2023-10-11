import os
import numpy as np
import time
import copy

from typing import Optional, Union

from .daqtask import DAQTask
from .ni_task import NITaskOutput

from ..generation_base import BaseGeneration

class NIGeneration(BaseGeneration):
    def __init__(self, task_name, signal=None, generation_name=None):
        """NI Generation class used for generating signals.
        
        Args:
            task_name (str, class object): Name of the task from NI Max or class object created with NITaskOutput() class using nidaqmx library.
            signal (numpy.ndarray): Signal to be generated. Shape is ``(n_samples, n_channels)`` or ``(n_samples,)``.
            generation_name (str, optional): Name of the generation class. Defaults to None, in which case the task name is used.
        """
        super().__init__()
        self.task_name = task_name
        
        if signal is not None:
            self.set_generation_signal(signal)
        
        self.task_terminated = True
        self.task_base = task_name
        
        if isinstance(task_name, str):
            self.NITask_used = False
        elif isinstance(task_name, NITaskOutput):
            self.NITask_used = True
        else:
            raise TypeError("task_name has to be a string or NITaskOutput object.")
        
        self.set_data_source(initiate=False)
        self.generation_name = task_name if generation_name is None else generation_name
        
    def set_generation_signal(self, signal):
        """sets signal that will be generated, and repeated in a loop.

        Args:
            signal (np.ndarray): numpy array with shape ``(n_samples, n_channels)`` or ``(n_samples,)``.
        """
        self.signal = signal
        if self.signal.ndim > 1:
            self.signal = self.signal.T

    def set_data_source(self, initiate=True):
        """Sets the data source for the generation.

        Args:
            initiate (bool, optional): intitiate NI task. Defaults to True.
        """
        if self.task_terminated:
            if self.NITask_used:
                channels_base = copy.deepcopy(self.task_base.channels)
                self.Task = NITaskOutput(self.task_base.task_name, self.task_base.sample_rate)
                self.task_name = self.task_base.task_name
                self.Task.channels = channels_base

            else:
                self.Task = DAQTask(self.task_base)
            
            self.task_terminated = False

        if self.NITask_used and initiate:
            if not hasattr(self.Task, 'task'):
                self.Task.initiate()
        
    def terminate_data_source(self):
        """Terminates the data source for the generation.
        """
        self.task_terminated = True
        self.clear_task()
    
    def generate(self):
        """Generates the signal.
        """
        if self.signal is None:
            raise ValueError("No signal set for generation.")
        self.Task.generate(self.signal, clear_task=False)

    def clear_task(self):
        """Clears NI output task.
        """
        if hasattr(self, 'Task'):
            self.Task.clear_task(wait_until_done=False)

            # generate zeros
            self.set_data_source()

            if self.signal.ndim == 1:
                zero_signal = np.zeros(self.signal.shape[0])
            else:
                zero_signal = np.zeros((self.signal.shape[0], 10))

            self.Task.generate(zero_signal, clear_task=False)
            self.Task.clear_task(wait_until_done=False)
            self.task_terminated = True
            
            del self.Task

    def run_generation(self):
        """Runs the signal generation.
        """
        self.is_running = True
        
        self.set_data_source()
        self.generate()