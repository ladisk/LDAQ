import os
import numpy as np

from .daqtask import DAQTask

class BaseGenerator:
    def __init__(self):
        self.is_running = True

    def generate(self):
        pass

    def run_generation(self):
        while self.is_running:
            self.generate()


class NIGenerator(BaseGenerator):
    def __init__(self, task_name, signal):
        super().__init__()
        self.task_name = task_name
        self.Task = DAQTask(self.task_name)
        self.signal = signal
    
    def generate(self):
        self.Task.generate(self.signal, clear_task=False)

    def clear_task(self):
        self.Task.clear_task(wait_until_done=False)

    def stop(self):
        self.clear_task()
        self.is_running = False