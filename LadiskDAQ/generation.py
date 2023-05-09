import os
import numpy as np
import time
import copy

from .daqtask import DAQTask
from .ni_task import NITaskOutput

class BaseGenerator:
    def __init__(self):
        self.is_running = True
        self.generation_name = "DefaultSignalGeneration"

    def generate(self):
        pass

    def run_generation(self):
        while self.is_running:
            self.generate()

    def set_data_source(self):
        pass

    def terminate_data_source(self):
        pass

    def stop(self):
        self.is_running = False
        self.terminate_data_source()


class NIGenerator(BaseGenerator):
    def __init__(self, task_name, signal, generation_name=None):
        """NI Generator class.
        
        :param task_name: Name of the task.
        :param signal: Signal to be generated. Shape is ``(n_samples, n_channels)`` or ``(n_samples,)``.
        """
        super().__init__()
        self.task_name = task_name
        self.signal = signal

        if self.signal.ndim > 1:
            self.signal = self.signal.T

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

    def set_data_source(self, initiate=True):
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
        self.task_terminated = True
        self.clear_task()
    
    def generate(self):
        self.Task.generate(self.signal, clear_task=False)

    def clear_task(self):
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

    def run_generation(self, run_time=None):
        self.is_running = True
        
        self.set_data_source()

        self.generate()
        