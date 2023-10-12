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
        self.delay = 0.0
        self.is_running = True
        self.generation_name = "DefaultSignalGeneration"

    def generate(self):
        """
        EDIT in child class. The child should call methods that generate the signal.
        """
        pass

    def run_generation(self, delay=None, block=False):
        """Runs generation. If block is True, the generation will block the main thread until generation is stopped.
        
        Args:
            delay (float, optional): Delay in seconds before generation starts. If None, no delay is added or
                                    previous delay is used. Defaults to None.
            block (bool, optional): If True, the generation will block the main thread until generation is stopped. 
                                    Defaults to False. 
        """
        if delay is not None:
            self.add_delay(delay)
        time.sleep(self.delay)
        
        self.set_data_source()
        if block:
            while self.is_running:
                self.generate()
        else:
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
        
    def add_delay(self, delay):
        """
        Adds delay before generation starts
        
        Args:
            delay (float): Delay in seconds
        """
        self.delay = delay

        