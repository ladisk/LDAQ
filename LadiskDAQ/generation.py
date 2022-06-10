import os
import numpy as np
import time

from .daqtask import DAQTask

class BaseGenerator:
    def __init__(self):
        self.is_running = True

    def generate(self):
        pass

    def run_generation(self):
        while self.is_running:
            self.generate()

    def set_data_source(self):
        pass

    def clear_data_source(self):
        pass

    def stop(self):
        self.is_running = False


class NIGenerator(BaseGenerator):
    def __init__(self, task_name, signal):
        super().__init__()
        self.task_name = task_name
        self.Task = DAQTask(self.task_name)
        self.signal = signal

    def set_data_source(self):
        if not hasattr(self, 'Task'):
            self.Task = DAQTask(self.task_name)
        
    def clear_data_source(self):
        self.clear_task()
    
    def generate(self):
        self.Task.generate(self.signal, clear_task=False)

    def clear_task(self):
        self.Task.clear_task(wait_until_done=False)
        del self.Task

    def run_generation(self):
        self.set_data_source()

        self.generate()

        while self.is_running:
            print('checking')
            time.sleep(0.5)
        
        self.clear_data_source()
        