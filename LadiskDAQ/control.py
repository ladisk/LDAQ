import time

class BaseControl:
    def __init__(self, acquisition, generation=None):
        self.acquisition = acquisition
        self.generattion = generation

    def run(self):
        while self.acquisition.is_running:
            self.run_control()
            time.sleep(0.01)

    def run_control(self):
        pass