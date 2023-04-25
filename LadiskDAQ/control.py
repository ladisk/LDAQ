"""Work in progress.
"""
import time

class BaseControl:
    def __init__(self, acquisition_names, generation_names, control_name=None):
        self.acquisition_names = acquisition_names
        self.generattion_names = generation_names
        self.control_name = control_name if control_name is not None else "DefaultControl"
        
    def _retrieve_core_object(self, core):
        """Retrieves core object to access acquisition and generation sources.
           This method is called by the Core() object and should not be called manually.

        Args:
            core (class): Core() object that handles all the data acquisition and generation.
        """
        self.core = core
    
    def control_init(self):
        """Should be edited in child class.
           This method is called before the control process is started.
        """
        pass
    def control_run(self):
        """Should be edited in child class. 
           This method is called periodically
           to control the generation and acquisition sources.
        """
        pass 
    def control_exit(self):
        """Should be edited in choild class. This method is called at the end when the control
           process is temrinated
        """
        pass

    def run_control(self):
        self.is_running = True
        
        self.control_function_init()
        while self.is_running:
            # perform control:
            self.control_function_run()
            time.sleep(0.01)
        self.control_function_exit()