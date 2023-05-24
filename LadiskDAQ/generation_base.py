import os
import numpy as np
import time
import copy

class BaseGeneration:
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

        