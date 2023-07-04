import numpy as np
import time
import multiprocessing as mp
import dill as pickle

from ..acquisition_base import BaseAcquisition

class SimulatedAcquisition(BaseAcquisition):
    """
    Simulated acquisition class that can be used when no source is present.
    """
    def __init__(self, acquisition_name=None):
        """
        Args:
            acquisition_name (str, optional): Name of the acquisition. Defaults to None, in which case the name "Simulator" is used.
        """
        super().__init__()
        
        self.acquisition_name = 'Simulator' if acquisition_name is None else acquisition_name

        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source
        
        self.child_process_started = False
        
    def __del__(self):
        """If class is deleted, stop the data generation process.
        """
        if self.child_process_started:
            self.stop_event.set()
            self.process.join()
            self.child_process_started = False
        
    def set_simulated_data(self, fun, channel_names=None, sample_rate=None, args=()):
        """sets simulated data to be returned by read_data() method. 
        This should also update self._channel_names_init list.
        
        NOTE: The function 'fun' should also include all library imports needed for its execution. This is due to serialization limitations of the function
        of 'dill' library in order to be able to pass the function to the child process. For example, if the function 'fun' uses numpy, it should be imported.

        Args:
            fun (function): function that returns numpy array with shape (n_samples, n_channels)
            channel_names (list, optional): list of channel names. Defaults to None, in which case the names "channel_0", "channel_1", ... are used.
            sample_rate (int, optional): sample rate of the simulated data. Defaults to None, in which case the sample rate of 1000 Hz is used.
            args (tuple, optional): arguments for the function. Defaults to ().
            
        Example:
        
        >>> def simulate_signal(t, f1, f2):
        >>>     import numpy as np
        >>>     sig1 = np.sin(2*np.pi*f1*t) + np.random.rand(len(t))*0.3
        >>>     sig2 = np.cos(2*np.pi*f2*t) + np.random.rand(len(t))*0.3
        >>>     return np.array([sig1, sig2]).T
        >>> 
        >>> acq_simulated = LDAQ.simulator.SimulatedAcquisition(acquisition_name='sim')
        >>> acq_simulated.set_simulated_data(simulate_signal, channel_names=["ch1", "ch2"], sample_rate=100000, args=(84, 120))
        >>> acq_simulated.run_acquisition(5.0)
            
        """
        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source

        self.simulated_function = fun
        self._channel_names_init = channel_names
        self.sample_rate = 1000 if sample_rate is None else sample_rate
        self._args = args
        
        time_array = np.arange(self.sample_rate)/self.sample_rate
        data = fun(time_array, *self._args)

        if data.ndim == 2:

            if channel_names is None:
                self._channel_names_init = [f"channel_{i}" for i in range(data.shape[1])]

            if data.shape[1] != len(self._channel_names_init):
                raise ValueError("Number of channels in data and channel_names does not match.")
        else:
            raise ValueError("Data must be 2D array.")
        
        self.set_data_source(initiate_data_source=True)
        self.set_trigger(1e20, 0)

    def set_simulated_video(self, fun, channel_name_video=None, sample_rate=None, args=()):
        """sets simulated video to be returned by read_data() method.
        This should also update self._channel_names_video_init and self._channel_shapes_video_init lists.

        NOTE: The function 'fun' should also include all library imports needed for its execution. This is due to serialization limitations of the function
        of 'dill' library in order to be able to pass the function to the child process. For example, if the function 'fun' uses numpy, it should be imported.
        
        Args:
            fun (function): function that takes time array as first argument and returns numpy array with shape (n_samples, width, height)
            channel_name_video (str, optional): name of the video channel. Defaults to None, in which case the name "video" is used.
            sample_rate (int, optional): sample rate of the simulated data. Defaults to None, in which case the sample rate of 30 Hz is used.
            args (tuple, optional): arguments for the function. Defaults to ().
        """
        self._channel_names_init         = [] # list of original data channels names from source 
        self._channel_names_video_init   = [] # list of original video channels names from source
        self._channel_shapes_video_init  = [] # list of original video channels shapes from source

        self.simulated_function = fun
        self.sample_rate = 30 if sample_rate is None else sample_rate
        self._args = args
        
        time_array = np.arange(self.sample_rate)/self.sample_rate
        data = fun(time_array, *self._args)

        if data.ndim == 3:
            if channel_name_video is None:
                self._channel_names_video_init = ["video_channel"]
            else:
                self._channel_names_video_init = [channel_name_video]

            self._channel_shapes_video_init = [data.shape[1:]]
        else:
            raise ValueError("Data must be 3D array.")
        
        self.set_data_source(initiate_data_source=True)
        self.set_trigger(1e20, 0)

        
    def set_data_source(self, initiate_data_source=True):
        """
        Initializes simulated data source
        """  
        if initiate_data_source:
            if not self.child_process_started:
                # Create a Pipe for communication between processes
                self.parent_conn, self.child_conn = mp.Pipe()
                # Event to signal stop of generation of simulated data:
                self.stop_event = mp.Event()
                
                # serialize function using pickle:
                ser_simulated_fun = pickle.dumps(self.simulated_function)
                
                self.child_process_started = True
                self.process = mp.Process(target=self.data_generator, args=(self.child_conn, self.stop_event, self.sample_rate, ser_simulated_fun, self._args))
                self.process.start()
                
                
            
        super().set_data_source()
        
    def terminate_data_source(self):
        """
        Terminates simulated data source
        """
        if self.child_process_started: # TODO: add logic to check if something has changed in the data source
                                       # if yes, then reset the data source, otherwise do not terminate it
            self.stop_event.set()
            # Wait for the process to finish
            self.process.join()
            
            self.child_process_started = False

    def read_data(self):
        """reads data from simulated data source.

        Returns:
            np.ndarray: data from serial port with shape (n_samples, n_channels).
        """
        time.sleep(0.002)
        # Send a request for data
        self.parent_conn.send('get_data')
        # Receive the data
        return self.parent_conn.recv()
    
    def clear_buffer(self):
        """
        Clears serial buffer.
        """
        self.read_data()
            
    def get_sample_rate(self):
        """Returns acquisition sample rate.

        Returns:
            float: estimated sample rate
        """
        
        return self.sample_rate
    
    @staticmethod
    def data_generator(connection, stop_event, sample_rate, ser_function, fun_args):
        """
        This function runs in a separate process and generates data (2D numpy arrays),
        and maintains a buffer of generated data.
        """
        print("setting up child process")
        import time
        import numpy as np
        
        #deserialize function using pickle:
        function = pickle.loads(ser_function)
        
        time_start = time.time()
        time_previous = time_start
        time_add = 0
        
        buffer = []
        print("child process set up")
        while not stop_event.is_set():
            time_now = time.time()
            time_elapsed = time_now - time_previous
            
            samples_to_read = int(time_elapsed * sample_rate)
            time_array = np.arange(samples_to_read)/sample_rate + time_add
            
            if len(time_array) == 0:
                continue
            
            data = function(time_array, *fun_args)
            
            if data.ndim == 3:
                data = data.reshape((-1, data.shape[1]*data.shape[2]))
                
            time_previous = time_now
            time_add = time_array[-1] + 1/sample_rate
            
            # Simulate data generation (using random numbers here)

            buffer.append(data)
            # Sleep for a bit to simulate time it takes to generate data
            time.sleep(0.01)

            # Check if there is a request for data
            if connection.poll():
                request = connection.recv()
                if request == 'get_data':
                    # Send the entire buffer as a numpy array
                    connection.send(np.vstack(buffer))
                    # Clear the buffer
                    buffer.clear()

    