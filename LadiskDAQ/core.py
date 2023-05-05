import time
import datetime
import os
import numpy as np
import keyboard
from beautifultable import BeautifulTable

import threading
import pickle


class Core():
    def __init__(self, acquisitions, generations=None, controls=None, visualization=None):
        """
        Initializes the Core instance by initializing its acquisition, generation, control and visualization sources. 

        :param acquisitions:  list of acquisition sources. If None, initializes as empty list.
        :param generations:   list of generation sources. If None, initializes as empty list.
        :param controls:      list of control sources. If None, initializes as empty list.
        :param visualization: visualization source. If None, initializes as empty.
        """
        acquisitions = [] if acquisitions is None else acquisitions
        generations  = [] if generations is None else generations
        controls     = [] if controls is None else controls
        
        self.acquisitions  = acquisitions if isinstance(acquisitions,list ) else [acquisitions]
        self.generations   = generations  if isinstance(generations, list ) else [generations]
        self.controls      = controls     if isinstance(controls,    list ) else [controls]
        self.visualization = visualization
        
        self.acquisition_names = [acq.acquisition_name for acq in self.acquisitions]
        self.generation_names = [gen.generation_name for gen in self.generations]
        if any(self.acquisition_names.count(s) > 1 for s in set(self.acquisition_names)): # check for duplicate acq. names
            raise Exception("Two or more acquisition sources have the same name. Please make them unique.")
        if any(self.generation_names.count(s) > 1 for s in set(self.generation_names)):
            raise Exception("Two or more generation sources have the same name. Please make them unique.")
        
        self.trigger_source_index = None
        
    def synchronize_acquisitions(self):
        """
        TODO: maybe there is a way to sync all acquisition sources.
        Maybe this function can be application specific.
        For example, maybe functional generator is attached to all sources at the same time
        and time shifts are calcualted.
        """
        pass
    
    
    def run(self, measurement_duration=None, autoclose=True, autostart=False, run_name="Run", save_interval=None, root='', verbose=2):
        """
        Runs the measurement with acquisition and generation sources that have already been set. This entails setting configuration
        and making acquiring and generation threads for each source, as well as starting and stopping them when needed. If visualization
        has been defined, it will run in a separate thread.

        :param measurement_duration: (float) measurement duration in seconds, from trigger event of any of the sources. 
                                            If None, the measurement runs forever until manually stopped. Default is None.
        :param autoclose: (bool) whether the sources should close automatically or not. Default is True.
        :param autostart: (bool): whether the measurement should start automatically or not. If True, start as soon as all the 
                                            acquisition sources are ready. Defaults to False.
        :param run_name: (str) name of the run. This name is used for periodic saving. Default is "Run".
        :param save_interval: (float) data is saved every 'save_periodically' seconds. Defaults to None,
                                    meaning data is not saved. The time stamp on measured data will equal to
                                    beginning of the measurement.
        :param root: (str) root directory where measurements are saved. Default is empty string.
        :param verbose: (int) 0 (print nothing), 1 (print status) or 2 (print status and hotkey legend). Default is 2.
        """
        self.run_name = run_name
        self.verbose  = verbose
        self.measurement_duration = measurement_duration
        self.save_interval = save_interval
        self.root = root
        self.autoclose = autoclose
        self.is_running_global = True
        self.autostart = autostart
        
        self.first = True # for printing trigger the first time.
        
        if self.visualization is None:
            self.keyboard_hotkeys_setup()
            if self.verbose == 2:
                self._print_table()
        else:
            self.verbose = 0
        
        if self.verbose in [1, 2]:
            print('\tWaiting for trigger...', end='')

        ####################
        # Thread setting:  #
        ####################
        
        self.lock = threading.Lock() # for locking a thread if needed.    
        self.triggered_globally = False
        self.thread_list = []

        # Make separate threads for data acquisition
        for acquisition in self.acquisitions:
            # update triggers from acquisition to match acquired samples to run_time:
            acquisition.is_standalone = False
            acquisition.reset_trigger()
            if self.measurement_duration is not None:
                acquisition.update_trigger_parameters(duration=self.measurement_duration, duration_unit="seconds")
            if autostart:
                acquisition.update_trigger_parameters(level=1e40)   
                
            thread_acquisition = threading.Thread(target=acquisition.run_acquisition )
            self.thread_list.append(thread_acquisition)

        # If generation is present, create generation thread
        for generation in self.generations:
            thread_generation  = threading.Thread(target=generation.run_generation )
            self.thread_list.append(thread_generation)

        for control in self.controls:
            thread_control = threading.Thread(target=control.run_control)
            self.thread_list.append(thread_control)
             
        # check events:
        thread_check_events = threading.Thread(target=self._check_events)
        self.thread_list.append(thread_check_events)
        
        # periodic data saving:
        if self.save_interval is not None:
            thread_periodic_saving = threading.Thread(target=self._save_measurement_periodically)
            self.thread_list.append(thread_periodic_saving)
            
        self.run_start_global = time.time()
        # start all threads:
        for thread in self.thread_list:
            thread.start()
        time.sleep(0.2)

        if self.visualization is not None:
            self.visualization.run(self)
        else:
            # Main Loop if no visualization:
            while self.is_running_global:
                time.sleep(0.5)

        # on exit:
        self.stop_acquisition_and_generation()
        
        for thread in self.thread_list:
            thread.join()
            
        if self.verbose in [1, 2]:
            print('Measurement finished.')
        
        if self.visualization is None:
            self.keyboard_hotkeys_remove()
    
    def _check_events(self):
        """
        The function _check_events checks for different events required to perform measurements. 
        It checks whether all acquisition and generation sources are running or not; if any of them are not running, 
        then it terminates the measurement. It also checks if any acquisition sources are triggered or if any additional 
        check functions added with add_check_events() method return True. If either of these conditions returns True, 
        it terminates the measurement. This function runs continuously in a separate thread until the is_running_global 
        variable is set to False.

        Args:
            None


        Returns:
            None
        """
        while self.is_running_global:
            acquisition_running = True
            if all(not acquisition.is_running for acquisition in self.acquisitions) and len(self.acquisitions) > 0:
                acquisition_running = False # end if all acquisitions are ended
            
            generation_running = True
            if all(not generation.is_running for generation in self.generations) and len(self.generations) > 0:
                generation_running = False
                
            control_running = True
            if all(not control.is_running for control in self.controls) and len(self.controls) > 0:
                control_running = False
                
            self.is_running_global = acquisition_running and generation_running and control_running
            
            # check that all acquisitions are ready:
            if not self.acquisitions[0].all_acquisitions_ready:
                all_acquisitions_ready = all(acq.is_ready for acq in self.acquisitions)
                if all_acquisitions_ready:
                    self.acquisitions[0]._all_acquisitions_ready()
                    if self.autostart:
                        self.start_acquisition()
            
            if any(acq.is_triggered() for acq in self.acquisitions) and not self.triggered_globally:
                self.triggered_globally = True
                
            if self.first and self.triggered_globally:
                if self.verbose in [1, 2]:
                    print()
                    print('triggered.') 
                    print('\tRecording...', end='') 
                self.first = False
                            
            # additional functionalities added with 'add_check_events()' method:   
            if hasattr(self, "additional_check_functions"):
                for fun in self.additional_check_functions:
                    if fun(self):
                        self.stop_acquisition_and_generation()
                                                
            time.sleep(0.05)   
            
    def add_check_events(self, *args):
        """
        Takes functions that takes only "self" argument and returns True/False. If any of the provided functions
        is True, the acquisition will be stopped. Each time this function is called, the previous additional
        check functions are erased.
        """
        self.additional_check_functions = []
        for fun in args:
            self.additional_check_functions.append(fun)   
                
    def set_trigger(self, source, channel, level, duration, duration_unit='seconds', presamples=0, trigger_type='abs'):
        """
        Sets trigger to one of the acquisition sources. 

        Args:
            trigger_source (int, str): Index (position in the 'acquisitions' list) or name of the acquisition source as a 
                                        string ('acquisition.acquisition_name') for which trigger is to be set. 

            trigger_channel (int): Channel number used for trigger.

            trigger_level (float): Trigger_level in Volts.

            duration (int): Duration of trigger source in terms of duration_unit defined (in seconds or samples).

            duration_unit (str): Unit of duration of trigger source. Can be 'seconds' or 'samples'. 

            presamples (int): Number of samples acquired before trigger.

            trigger_type (str): Type of the trigger. Can be 'abs' or 'edge'.

        Returns: 
            None. 

        Raises: 
            KeyError: Invalid duration unit specified. Only 'seconds' and 'samples' are possible. 

        Other requirements: 
            Expect delay between different acquisition sources due to unsynchronized sources. 

        """
        if duration_unit=="samples":
            duration = int(duration)
            
        # set external trigger option to all acquisition sources:
        if type(source) == str:
            source = self.acquisition_names.index(source)
        self.trigger_source_index = source # save source index on which trigger is set
        
        for idx, acq in enumerate(self.acquisitions):
            if idx == source: #set trigger
                acq.set_trigger(
                    level=level, 
                    channel=channel, 
                    duration=duration, 
                    duration_unit=duration_unit, 
                    presamples=presamples, 
                    type=trigger_type
                )
            else:
                source_sample_rate = self.acquisitions[source].sample_rate
                presamples_seconds = presamples/source_sample_rate
                presamples_other   = int(presamples_seconds*acq.sample_rate)
                
                if duration_unit == "seconds":
                    duration_seconds = duration
                    acq.update_trigger_parameters(duration=duration_seconds, duration_unit="seconds", presamples=presamples_other)
                elif duration_unit == "samples": # if specified as samples convert to seconds for other acquisition sources.
                    duration_seconds = duration/source_sample_rate
                    duration_samples = int(duration_seconds*acq.sample_rate)
                    acq.update_trigger_parameters(duration=duration_samples, duration_unit="samples", presamples=presamples_other)
                    
                else:
                   raise KeyError("Invalid duration unit specified. Only 'seconds' and 'samples' are possible.")
            
    def keyboard_hotkeys_setup(self):
        """Adds keyboard hotkeys for interaction.
        """
        id1 = keyboard.add_hotkey('s', self.start_acquisition)
        id2 = keyboard.add_hotkey('q', self.stop_acquisition_and_generation)
        self.hotkey_ids = [id1, id2]
        
    def keyboard_hotkeys_remove(self):
        """Removes all keyboard hotkeys defined by 'keyboard_hotkeys_setup'.
        """
        for id in self.hotkey_ids:
            keyboard.remove_hotkey(id)
            
    def stop_acquisition_and_generation(self):
        """Stops all acquisition and generation sources.
        """
        for acquisition in self.acquisitions:
            acquisition.stop()
        for generation in self.generations:
            generation.stop()
            
    def start_acquisition(self):
        """Starts acquisitions sources.
        """
        if not self.triggered_globally:
            self.triggered_globally = True
            
            # 1 acq source triggers others through CustomPyTrigger parent class
            with self.acquisitions[0].lock_acquisition: 
                self.acquisitions[0].activate_trigger()
    
    def _print_table(self):
        """Prints the table of the hotkeys of the application to the console.
        The table contains the hotkeys, as well as a short description of each
        hotkey. The table is printed using the BeautifulTable library.
        """
        table = BeautifulTable()
        table.rows.append(["s", "Start the measurement manually (ignore trigger)"])
        table.rows.append(["q", "Stop the measurement"])
        table.columns.header = ["HOTKEY", "DESCRIPTION"]
        print(table)
     
    def get_measurement_dict_PLOT(self):
        """
        Returns only NEW acquired data from all sources.
        NOTE: This function is used for plotting purposes only.
              Other functions should use 'get_measurement_dict(N_seconds="new")' instead.
        """
        new_data_dict = {}
        for idx, acq in enumerate(self.acquisitions):
            # retireves new data from this source
            new_data_dict[self.acquisition_names[idx]] = acq.get_data_PLOT() 
        return new_data_dict
    
    def get_measurement_dict(self, N_seconds=None):
        """Returns measured data from all sources.

        Args:
            N_seconds (float, str, optional): last number of seconds of the measurement. 
                        if "new" then only new data is returned. Defaults to None.

        Returns:
            dict: Measurement dictionary
            """        
        self.measurement_dict = {}
        for idx, name in enumerate(self.acquisition_names):
            if N_seconds is None:
                N_points = None
            elif type(N_seconds)==float or type(N_seconds)==int:
                N_points = int( N_seconds * self.acquisitions[idx].sample_rate ) 
            elif N_seconds=="new":
                N_points = N_seconds
            else:
                raise KeyError("Wrong argument type passed to N_seconds.")
                
            self.measurement_dict[ name ] = self.acquisitions[idx].get_measurement_dict(N_points)
        
        return self.measurement_dict    
    
    def save_measurement(self, name=None, root=None, timestamp=True, comment=None):
        """Save acquired data from all sources into one dictionary saved as pickle.
        
        :param name: filename, if None filename defaults to run name specified in run() method.
        :param root: directory to save to
        :param timestamp: include timestamp before 'filename'
        :param comment: comment on the saved file
        """
        if name is None:
            name = self.run_name
        if root is None:
            root = self.root
            
        self.measurement_dict = self.get_measurement_dict()
        if comment is not None:
            self.measurement_dict['comment'] = comment
            
        if not os.path.exists(root):
            os.mkdir(root)

        if timestamp:
            now = datetime.datetime.now()
            stamp = f'{now.strftime("%Y%m%d_%H%M%S")}_'
        else:
            stamp = ''

        filename = f'{stamp}{name}.pkl'
        path = os.path.join(root, filename)
        pickle.dump(self.measurement_dict, open(path, 'wb'), protocol=-1)   
        
    def _save_measurement_periodically(self):
        """Periodically saves the measurement data."""
        name = self.run_name
        root = self.root
        
        start_time = time.time()
        file_created = False     
            
        running = True
        delay_saving = 0.5 # seconds
        delay_start  = time.time()
        
        while running:            
            time.sleep(0.2)  
                      
            # implemented time delay:
            if self.is_running_global:
                delay_start = time.time()
            elif time.time()-delay_start > delay_saving:
                running = False
            else:
                pass
            
            # periodic saving: 
            if self.triggered_globally:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.save_interval:
                    start_time = time.time()
                    
                    if not file_created:
                        now = datetime.datetime.now()
                        file_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{name}.pkl"
                        file_created = True
                        
                    self._open_and_save(file_name, root)
                    
        time.sleep(0.5)
        self._open_and_save(file_name, root)
                        
    def _open_and_save(self, file_name, root):
        """Open existing file and save new data."""
        file_path = os.path.join(root, file_name)
        # Load existing data
        if os.path.exists(file_path):
            data = load_measurement(file_name, root)
        else:
            data = {}

        # Update data with new measurements
        for acq in self.acquisitions:
            name = acq.acquisition_name
            if acq.is_triggered():
                measurement = acq.get_measurement_dict(N_points = "new")
                
                if name not in data:
                    data[name] = measurement
                else:
                    new_data = measurement['data']
                    
                    if len(data[name]['time']) > 0:
                        time_last = data[name]['time'][-1]
                    else:
                        time_last = 0    
                        
                    new_time = measurement['time'] + time_last + 1/acq.sample_rate
                    
                    data[name]['data'] = np.concatenate((data[name]['data'], new_data), axis=0)
                    data[name]['time'] = np.concatenate((data[name]['time'], new_time), axis=0)

        # Save updated data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=-1)     
            print("saved.")  

# open measurements:
def load_measurement(name, directory='' ):
    """
    Loads a measurement from a pickle file.
    """
    with open(directory+'/' + name, 'rb') as f:
        return pickle.load(f)
