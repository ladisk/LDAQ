import os
import numpy as np
import time
import copy

class BaseGeneration:
    """
    Base class for signal generation. Used for generating signals.
    """
    def __init__(self):
        """
        EDIT in child class. The child class should call super().__init__() and set the following attributes:
        - self.generation_name
        """
        self.is_running = True
        self.generation_name = "DefaultSignalGeneration"

    def generate(self):
        """
        EDIT in child class. The child should call methods that generate the signal.
        """
        pass

    def run_generation(self):
        while self.is_running:
            self.generate()

    def set_data_source(self):
        """
        EDIT in child class. The child should call methods that set the signal.
        """
        pass

    def terminate_data_source(self):
        """
        EDIT in child class. The child should call methods that terminates and exits signal generation device correctly.
        """
        pass

    def stop(self):
        """
        Stops the generation.
        """
        self.is_running = False
        self.terminate_data_source()

        