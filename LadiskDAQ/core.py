import enum
from msilib.schema import Error
from re import X
import time
import datetime
import os
from xml.sax import SAXException
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from beautifultable import BeautifulTable

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys

from pyTrigger import pyTrigger

import threading
import types
import pickle


AXIS_SCALES       = ['logx', 'logy']
INBUILT_FUNCTIONS = ['fft', 'frf_amp', 'frf_phase']


class Core():
    def __init__(self, acquisitions, generations=None, controls=None, visualization=None):
        acquisitions = [] if acquisitions is None else acquisitions
        generations  = [] if generations is None else generations
        controls     = [] if controls is None else controls
        
        self.acquisitions  = acquisitions if isinstance(acquisitions,list ) else [acquisitions]
        self.generations   = generations  if isinstance(generations, list ) else [generations]
        self.controls      = controls     if isinstance(controls,    list ) else [controls]
        self.visualization = visualization
        
        self.acquisition_names = [acq.acquisition_name for acq in self.acquisitions]
        if any(self.acquisition_names.count(s) > 1 for s in set(self.acquisition_names)): # check for duplicate acq. names
            raise Exception("Two or more acquisition sources have the same name. Please make them unique.")

    def synchronize_acquisitions(self):
        """
        TODO: maybe there is a way to sync all acquisition sources.
        Maybe this function can be application specific.
        For example, maybe functional generator is attached to all sources at the same time
        and time shifts are calcualted.
        """
        pass
    
    
    def run(self, measurement_duration=None, run_name="Run", save_interval=None, root='', verbose=2):
        """
        :param measurement_duration: measurement duration in seconds, from trigger event of any of the sources. 
                         If None the measurement runs for ever until manually stopped.
        :param save_interval: (float) data is saved every 'save_periodically' seconds. Defaults to None,
                                    meaning data is not saved. The time stamp on measured data will equal to
                                    beggining of the measurement.
        :param run_name: name of the run. This name is used for periodic saving
        :param verbose: 0 (print nothing), 1 (print status) or 2 (print status and hotkey legend). 
        """
        self.run_name = run_name
        self.verbose  = verbose
        self.measurement_duration = measurement_duration
        self.save_interval = save_interval
        self.root = root
        self.is_running_global = True
        
        self.keyboard_hotkeys_setup()
        if verbose == 2:
            self._print_table()
        
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
            if self.measurement_duration is not None:
                acquisition.update_trigger_parameters(duration=self.measurement_duration, duration_unit="seconds")   
            thread_acquisition = threading.Thread(target=acquisition.run_acquisition )
            self.thread_list.append(thread_acquisition)

        # If generation is present, create generation thread
        for generation in self.generations:
            thread_generation  = threading.Thread(target=generation.run_generation )
            self.thread_list.append(thread_generation)

        for control in self.controls:
            thread_control = threading.Thread(target=control.run)
            self.thread_list.append(thread_control)
             
        # check events:
        thread_check_events = threading.Thread(target=self._check_events)
        self.thread_list.append(thread_check_events)
        
        # if global trigger is set run the trigger in separate thread:
        if hasattr(self, "global_Trigger"):
            self.set_global_trigger(self.trigger_source, self.trigger_channel, 
                                    self.trigger_level,   self.trigger_type)
            
            thread_global_trigger = threading.Thread(target=self._run_global_trigger)
            self.thread_list.append(thread_global_trigger)
           
        # periodic data saving:
        if self.save_interval is not None:
            thread_periodic_saving = threading.Thread(target=self._save_measurement_periodically)
            self.thread_list.append(thread_periodic_saving)
            
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
            
        self.keyboard_hotkeys_remove()
    
    def _check_events(self):
        """
        Checks for:
        - if all acquisition and generation sources are running. If one of them is not, then measurement terminated
        - if any acquisition source has been triggered
        - if any additional check functions added with 'add_check_events()' return True. If so, measurement is terminated
        
        Executed in a separate thread.
        """
        while self.is_running_global:
            acquisition_running = True
            if all(not acquisition.is_running for acquisition in self.acquisitions) and len(self.acquisitions) > 0:
                acquisition_running = False # end if all acquisitions are ended
            
            generation_running = True
            if all(not generation.is_running for generation in self.generations) and len(self.generations) > 0:
                generation_running = False
                
            self.is_running_global = acquisition_running and generation_running
            
            # Check if any acquisition sources are triggered:
            if not self.triggered_globally: # only if measurement is not already running
                with self.lock:
                    for acquisition in self.acquisitions:
                        if acquisition.is_triggered():
                            self.start_acquisition()
                            break
                        else:
                            pass
                            
            # additional functionalities added with 'add_check_events()' method:   
            if hasattr(self, "additional_check_functions"):
                for fun in self.additional_check_functions:
                    if fun(self):
                        self.stop_acquisition_and_generation()
                                                
            time.sleep(0.05)      

    def set_global_trigger(self, source, channel, level, trigger_type='abs'):
        """Sets global trigger of the acquisition. NOTE: since acquisition sources may not be synchronized,
        expect up to 0.5 second delay between different acquisition sources. 

        Args:
            trigger_source (int, str): acquisition source index (position in the 'acquisitions' list) or
                                       name of the acquisition source as string (acquisition.acquistion_name)
            trigger_channel (int): channel number used for trigger
            trigger_level (float): trigger_level
        """
        # set external trigger option to all acquisition sources:
        for acq in self.acquisitions:
            acq.set_external_trigger()
        
        # set up global trigger:
        if type(source) == int:
            acquisition_trigger = self.acquisitions[source]
            
        elif type(source) == str:
            for idx, acq in enumerate(self.acquisitions):
                if source == acq.acquisition_name:
                    acquisition_trigger = acq
                    source = idx
                    break
            if source is None:
                raise Exception("Trigger source name was not found in any of the acquisition sources.")
        else:
            raise TypeError("trigger_source type can only be index or string.")
        
        self.trigger_source  = source
        self.trigger_channel = channel
        self.trigger_level   = level
        self.trigger_type    = trigger_type

        duration = 5. # does not matter, since this pyTrigger is only used for triggering not data storing
        self.global_trigger_buffer_samples = int( duration * acquisition_trigger.sample_rate )

        
        self.global_Trigger = pyTrigger(
                rows= self.global_trigger_buffer_samples, 
                channels=1, # only the triggering chanel will be passed to this pyTrigger instance
                trigger_type=self.trigger_type,
                trigger_channel=0, 
                trigger_level=self.trigger_level,
                presamples=0)
        
    def _run_global_trigger(self):
        time.sleep(0.5)
        while not self.triggered_globally:
            with self.lock:
                _, data = self.acquisitions[self.trigger_source].get_data()
            data = data[-self.global_trigger_buffer_samples:, self.trigger_channel:self.trigger_channel+1]
            self.global_Trigger.add_data(data)
            
            if self.global_Trigger.triggered:
                self.start_acquisition()
                
    def set_trigger(self, source, channel, level, duration, duration_unit='seconds', presamples=0, trigger_type='abs'):
        """Sets trigger to one of acquisition sources. NOTE: since acquisition sources may not be synchronized,
        expect delay between different acquisition sources. 

        Args:
            trigger_source (int, str): acquisition source index (position in the 'acquisitions' list) or
                                       name of the acquisition source as string (acquisition.acquistion_name)
            trigger_channel (int): channel number used for trigger
            trigger_level (float): trigger_level
        """
        # set external trigger option to all acquisition sources:
        if type(source) == str:
            source = self.acquisition_names.index(source)
        
        for idx, acq in enumerate(self.acquisitions):
            if idx == source: #set trigger
                acq.set_internal_trigger()
                acq.set_trigger(
                    level=level, 
                    channel=channel, 
                    duration=duration, 
                    duration_unit=duration_unit, 
                    presamples=presamples, 
                    type=trigger_type
                )
            else:
                duration_seconds = duration
                if duration_unit == "samples": # if specified as samples convert to seconds for other acquisition sources.
                    source_sample_rate = self.acquisitions[source].sample_rate
                    duration_seconds = duration*source_sample_rate
                    
                acq.update_trigger_parameters(duration=duration_seconds, duration_unit="seconds")
                acq.set_external_trigger()
            
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
            
            for acquisition in self.acquisitions:
                with acquisition.lock_acquisition: 
                    acquisition.clear_buffer()
                    acquisition.trigger()
            print('triggered.') 
            print('\tRecording...', end='') 
    
    def _print_table(self):
        table = BeautifulTable()
        table.rows.append(["s", "Start the measurement manually (ignore trigger)"])
        table.rows.append(["q", "Stop the measurement"])
        table.columns.header = ["HOTKEY", "DESCRIPTION"]
        print(table)
        
    def add_check_events(self, *args):
        """
        Takes functions that takes only "self" argument and returns True/False. If any of the provided functions
        is True, the acquisition will be stopped. Each time this function is called, the previous additional
        check functions are erased.
        """
        self.additional_check_functions = []
        for fun in args:
            self.additional_check_functions.append(fun)
     
    def get_measurement_dict(self, N_seconds=None):
        """Returns measured data from all sources

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
        """Periodically save acquired data from all sources."""
        name = self.run_name
        root = self.root
        
        start_time = time.time()
        file_created = False
        file_path = None      
            
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
                        file_path = os.path.join(root, file_name)
                        file_created = True
                        
                    # Load existing data
                    if os.path.exists(file_path):
                        data = load_measurement(file_name, root)
                    else:
                        data = {}

                    # Update data with new measurements
                    for acq in self.acquisitions:
                        name = acq.acquisition_name
                        #with acq.lock_acquisition:
                       
                        measurement = acq.get_measurement_dict(N_points = "new")
                    
                        #time_ = measurement['time']
                        #if time_.shape[0] >0:
                        #    last = time_[-1]
                        #else:
                        #    last = 0
            
                        if name not in data:
                            data[name] = measurement
                        else:
                            new_data = measurement['data']
                            
                            if len(data[name]['time']) > 0:
                                time_last = data[name]['time'][-1]
                            else:
                                time_last = 0    
                            #print("core", name, time_last, last)
                                
                            new_time = measurement['time'] + time_last + 1/acq.sample_rate
                            
                            data[name]['data'] = np.concatenate((data[name]['data'], new_data), axis=0)
                            data[name]['time'] = np.concatenate((data[name]['time'], new_time), axis=0)

                    # Save updated data
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=-1)       
                        
    
class LDAQ():
    """Visualization and triggering."""
    def __init__(self, acquisition, generation=None, control=None):
        """
        :param acquisition: object created with one of the acquisition classes.
        :param generation: optional, object created with one of the generation classes.
        """

        self.acquisition = acquisition
        self.generation  = generation

        if control:
            if self.generation:
                self.control = control(self.acquisition, self.generation)
            else:    
                self.control = control(self.acquisition)
        else:
            self.control = None

        # plot window settings:
        self.configure()

        # store any temporary variables into this list
        self.temp_variables = []

    def configure(self, plot_layout='default', max_time=5.0, nth_point='auto', autoclose=True, refresh_interval=0.01, show_live_plot=True):
        """Configure the plot window settings.
        
        :param plot_layout: layout of the plots and channels. "default" or dict. Keys of dict are (axis 0, axis 1) of the subplot
            layout, values are lists of channel indices to show on the subplots. See examples below.
        :param max_time: max time to show on the plot.
        :param nth_point: Only show every n_th point on the live plot.
        :param autoclose: Autoclose the window after the measurement is done.
        
        Plot layout
        -----------
        NORMAL TIME PLOTS: With plot layout, the user can define on which subplot the channels will be plotted. An example of
        plot layout is:
        >>> plot_layout = {
            (0, 0): [0, 1],
            (0, 1): [2, 3]
        }

        On first subplot (0, 0), channels 0 and 1 will be plotted, on the second subplot (0, 1), channels 2 and 3 will be plotted.
        Channels can also be plotted one against another. 
        
        CHANNEL vs. CHANNEL PLOT: If, for example, we wish to plot channel 1 as a function of channel 0, input
        channel indices in a tuple; first the channel to be plotted on x-axis, and second the channel to be plotted on y-axis:
        >>> plot_layout = {
            (0, 0): [0, 1],
            (0, 1): [2, 3],
            (1, 0): [(0, 1)]
        }

        FOURIER TRANSFORM PLOT: The DFT of the signal can be computed on the fly. To define the subplots and channels where the FFT is computed, 
        add "fft" as an element into channel list. Additionaly 'logy' and 'logx' scalings can be set:
        >>> plot_layout = {
            (0, 0): [0, 1],               # Time series
            (0, 1): [2, 3],               # Time series
            (1, 0): [(0, 1)],             # ch1 = f( ch0 )
            (1, 1): [2, 3, "fft", "logy"] # FFT(2) & FFT(3), log y scale
        }

        CUSTOM FUNCTION PLOT: Lastly, the signals can be modified for visualization by specifying a custom function, that is passed to the channel list.
        Example below computes the square of the signal coming from channels 1 and 2. 
        >>> plot_layout = {
            (0, 0): [0, 1],               # Time series
            (0, 1): [2, 3],               # Time series
            (1, 0): [(0, 1)],             # ch1 = f( ch0 )
            (1, 1): [2, 3, fun]           # fun(2) & fun(3)
        }

        function definition example:

        def fun(self, channel_data):
            '''
            :param self:         instance of the acquisition object (has to be there so the function is called properly)
            :param channel_data: channel data
            '''
            return channel_data**2

        CUSTOM FUNCTION PLOT (channel vs. channel): 
        >>> plot_layout = {
            (0, 0): [(0, 1), fun]         # 2Darray = fun( np.array([ch0, ch1]).T )
        }
    	
        function definition examples:

        def fun(self, channel_data):
            '''
            :param self:         instance of the acquisition object (has to be there so the function is called properly)
            :param channel_data: 2D channel data array of size (N, 2)

            :return: 2D array np.array([x, y]).T that will be plotted on the subplot.
            '''
            ch0, ch1 = channel_data.T

            x =  np.arange(len(ch1)) / self.acquisition.sample_rate # time array
            y = ch1**2 + ch0 - 10

            return np.array([x, y]).T

         def fun(self, channel_data):
            '''
            :param self:         instance of the acquisition object (has to be there so the function is called properly)
            :param channel_data: 2D channel data array of size (N, 2)

            :return: 2D array np.array([x, y]).T that will be plotted on the subplot.
            '''
            ch0, ch1 = channel_data.T

            x = np.arange(len(ch0)) / self.acquisition.sample_rate # time array
            y = ch1 + ch0 # sum up two channels

            # ---------------------------------------
            # average across whole acquisition:
            # ---------------------------------------
            # ensure number of samples is the same and perform averaging:
            if len(ch0) == int(self.max_samples): # at acquisition start, len(ch0) is less than self.max_samples
                
                # create class variables:
                if not hasattr(self, 'var_y'):
                    self.var_y = y
                    self.var_x = x
                    self.var_i = 0

                    # these variables will be deleted from LDAQ class after acquisition run is stopped: 
                    self.temp_variables.extend(["var_y", "var_x", "var_i"]) 
                
                self.var_y = (self.var_y * self.var_i + y) / (self.var_i + 1)
                self.var_i += 1

                return np.array([self.var_x, self.var_y]).T

            else:
                return np.array([x, y]).T
        """
        self.show_live_plot = show_live_plot
        self.plot_channel_layout = plot_layout
        if self.plot_channel_layout == "default":
            self.plot_channel_layout = {(0, 0):  [i for i in range(len(self.acquisition.channel_names)) ] }

        self.maxTime          = max_time
        self.autoclose        = autoclose
        self.refresh_interval = refresh_interval

        if type(nth_point) == int:
            self.nth_point = nth_point
        elif nth_point == 'auto':
            self.nth_point = auto_nth_point(self.plot_channel_layout, max_time, self.acquisition.sample_rate, max_points_to_refresh=1e5)
        
        self.max_samples = int(self.maxTime*self.acquisition.sample_rate) # max samples to display on some plots based on self.maxTime        

    def run(self, verbose=2):
        """
        :param verbose: 0 (print nothing), 1 (print status) or 2 (print status and hotkey legend). 
        """
        self.verbose = verbose
        if not self.show_live_plot:
            self.verbose = 2

        if verbose == 2:
            table = BeautifulTable()
            table.rows.append(["q", "Stop the measurement"])
            table.rows.append(["s", "Start the measurement manually (without trigger)"])
            table.rows.append(["f", "Freeze the plot during the measurement"])
            table.rows.append(["Space", "Resume the plot after freeze"])
            table.columns.header = ["HOTKEY", "DESCRIPTION"]
            print(table)
        
        if self.verbose in [1, 2]:
            print('\tWaiting for trigger...', end='')

        self.acquisition_started = False

        self.thread_list = []

        # Make separate threads for data acquisition
        thread_acquisition = threading.Thread(target=self.acquisition.run_acquisition )
        self.thread_list.append(thread_acquisition)

        # If generation is present, create generation thread
        if self.generation != None:
            thread_generation  = threading.Thread(target=self.generation.run_generation )
            self.thread_list.append(thread_generation)

        if self.control:
            thread_control = threading.Thread(target=self.control.run)
            self.thread_list.append(thread_control)

        # initialize plot window:
        if self.show_live_plot:
            self.plot_window_init()

        self.FREEZE_PLOT = False

        # start both threads:
        for thread in self.thread_list:
            thread.start()
        time.sleep(0.1)

        # while data is being generated and collected:
        while self.is_running():
            time.sleep(self.refresh_interval)

            self._check_events()

            # update plot window:
            if not self.FREEZE_PLOT and self.show_live_plot:
                self.plot_window_update()
            else:
                if not self.acquisition_started and self.acquisition.Trigger.triggered:
                    print('triggered.') 
                    print('\tRecording...', end='')
                    self.acquisition_started = True


    def is_running(self):
        """
        Function that checks whether the acquisition or generation class are running.
        """
        if self.generation == None:
            running = self.acquisition.is_running
        else:
            running = self.acquisition.is_running and self.generation.is_running
        
        if not running:
            self.acquisition.stop()
            if self.generation != None:
                self.generation.stop()

            if self.verbose in [1, 2]:
                print('stop.')
            
            # after DAQ is completed, join the threads with the main thread:
            for thread in self.thread_list:
                thread.join()

            if self.show_live_plot:
                self.plot_window_exit() # waits for plot window to be closed.
            self.clear_temp_variables()

        return running

    def _check_events(self):
        """
        Function that would disable DAQ, for example keyboard presses
        """
        if keyboard.is_pressed('q'):
            self.acquisition.stop()
            if self.generation != None:
                self.generation.stop()
        
        if keyboard.is_pressed('f'):
            self.FREEZE_PLOT = True

        if keyboard.is_pressed('Space'):
            self.FREEZE_PLOT = False

        if keyboard.is_pressed('s'):
            self.acquisition.Trigger.triggered = True
            
        if hasattr(self, "additional_check_functions"):
            for fun in self.additional_check_functions:
                if fun(self):
                    self.acquisition.stop()
        
    
    def add_check_events(self, *args):
        """
        Takes functions that takes only "self" argument and returns True/False. If any of the provided functions
        is True, the acquisition will be stopped. Each time this function is called, the previous additional
        check functions are erased.
        """
        self.additional_check_functions = []
        for fun in args:
            self.additional_check_functions.append(fun)

    def _create_plot(self, channels, pos_x, pos_y, label_x="", label_y="", unit_x="", unit_y=""):
        # subplot options:
        p = self.win.addPlot(row=pos_x, col=pos_y) 
        p.setLabel('bottom', label_x, unit_x)
        p.setLabel('left', label_y, unit_y)
        # p.setDownsampling(mode='subsample', auto=True)
        p.addLegend()
        p.showGrid(x = True, y = True, alpha = 0.3)  

        curves = self._create_curves(p, channels)

        return p, curves

    def _create_curves(self, plot, channels):
        """
        channels - index
        """
        color_dict = {ch:ind for ind, ch in enumerate(self.acquisition.channel_names)}
        curves = []
        for i, channel in enumerate(channels):
            if type(channel) == tuple:
                channel_name_x = self.acquisition.channel_names[channel[0]]
                channel_name_y = self.acquisition.channel_names[channel[1]]
                label = f"{channel_name_y} = f({channel_name_x})"
            else:
                channel_name_x = "Time"
                channel_name_y = self.acquisition.channel_names[channel]
                label = channel_name_y

            color = color_dict[ channel_name_y ]
            curve = plot.plot( pen=pg.mkPen(color, width=2), name=label) 
            
            # initialize some data:
            if type(channel) == tuple:
                curve.setData(self.dummy_data, self.dummy_data)
            else:
                curve.setData(self.dummy_data)

            curve.setPos(0, 0)
            curves.append(curve)
        return curves

    def plot_window_init(self):
        """
        Initializes plot window.
        """
        self.dummy_data = np.zeros(self.max_samples ) # dummy data

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsLayoutWidget(show=True) # creates plot window
        self.win.setWindowTitle('Measurement Monitoring')
        self.win.resize(800,400)

        # create window layout:
        self.curves_dict = {}
        self.plot_dict = {}
        self.fun_dict = {}
        positions = list(self.plot_channel_layout.keys())
        positions = [positions[i] for i in np.lexsort(np.array(positions).T[::-1])] # sort plot positions

        for j, (pos_x, pos_y) in enumerate(positions):

            # extract info from channel layout dict:
            settings = self.plot_channel_layout[(pos_x, pos_y)]
            channels = [i for i in settings if ((type(i)==int)or(type(i)==tuple))]
            #print(channels)
            scalings = [i for i in settings if i in AXIS_SCALES]
            functions = [i for i in settings if ((i in INBUILT_FUNCTIONS) or (type(i) == types.FunctionType))] # check if fun string or custom function
            
            # handle any additional functions for channel processing
            if len(functions) == 0:
                function = lambda self, x: x
                function_name = "none"

            elif len(functions) == 1:
                function = functions[0]

                if type(function) == str:
                    if function == "fft":
                        function_name = "fft"
                        function = _fun_fft
                    elif function == "frf_amp":
                        function_name = "frf_amp"
                        function = _fun_frf_amplitude
                    elif function == "frf_phase":
                        function_name = "frf_phase"
                        function = _fun_frf_phase
                    else:
                        pass
                else:
                    function_name = "custom"

            else:
                raise NotImplementedError("Only 1 custom function per subplot is supported.")

            self.fun_dict[(pos_x, pos_y)]    = function

            # create subplot and curves on the subplot:
            plot, curves = self._create_plot(channels=channels, pos_x=pos_x, pos_y=pos_y, label_x="", label_y="", unit_x="", unit_y="")

            # ADDITIONAL PLOT SETTINGS:                
            # channel vs. channel plot
            if type(channels[0]) == tuple:     # if channel vs. channel
                pass
            
            elif len(function(self, np.ones(10)).shape) > 1: # if function returns 2D array, then auto x range is needed
                pass

            # normal time signals channel:
            else:                                # if time signal
                plot.setRange(xRange=[-self.maxTime, 0]) 

            # set scalings:
            if len(scalings) > 0:
                for scaling in scalings:
                    logx = False
                    logy = False
                    if scaling == "logx":
                        logx = True
                    elif scaling == "logy":
                        logy = True
                    else:
                        pass
                    plot.setLogMode(x=logx, y=logy)
            else:
                 plot.setLogMode(x=False, y=False)

            # built in functions:
            if function_name == "fft" or function_name == "frf_amp" or function_name == "frf_phase":
                plot.setRange(xRange=[0, self.acquisition.sample_rate/6])


            self.curves_dict[(pos_x, pos_y)] = curves
            self.plot_dict[(pos_x, pos_y)]   = plot

        # ncols = len(set( [pos[1] for pos in positions ])  )
        # nrows = len(set( [pos[0] for pos in positions ])  )
        # for row in range(nrows):
        #     self.win.ci.layout.setRowStretchFactor(row, 1)
        # for ncols in range(ncols):
        #     self.win.ci.layout.setColumnStretchFactor(1, 1)

    def plot_window_update(self):
        """
        Updates plot window with collected data.
        """
        self.acquisition.plot_data = self.acquisition.plot_data[-int(self.max_samples*1.5):]

        data = self.acquisition.plot_data[-self.max_samples:]
        # create fixed time array:
        self.time_arr = -1*(np.arange(data.shape[0])/self.acquisition.sample_rate)
        #data = data[::self.nth_point]

        if not self.acquisition_started and self.acquisition.Trigger.triggered:
            self.win.setBackground('lightgreen')
            if self.verbose in [1, 2]:
                print('triggered.') 
                print('\tRecording...', end='')
            self.acquisition_started = True

        for position in self.plot_channel_layout:
            channels = self.plot_channel_layout[ position ]
            curves = self.curves_dict[ position ]
            function = self.fun_dict[ position ]

            for channel, curve in zip(channels, curves):
                if type(channel) == tuple: # plotting channel vs. channel
                    channel_x, channel_y = channel
                    fun_return = function(self, data[:, [channel_x, channel_y]])
                    x_values, y_values = fun_return.T

                    if len(y_values) == len(data[:, channel_y]):
                        x_values = x_values[::self.nth_point]
                        y_values = y_values[::self.nth_point]

                else:
                    y_values = data[:, channel]
                    fun_return = function(self, y_values)

                    if len(fun_return.shape) == 1: # if function returns only 1D array
                        y_values = fun_return[::self.nth_point]
                        x_values = self.time_arr[::-self.nth_point] # if time array then do not use function_x
                    else:  # function returns 2D array (e.g. fft returns freq and amplitude)
                        x_values, y_values = fun_return.T # expects 2D array to be returned
                        if len(y_values) == len(data[:, channel]):
                            x_values = x_values[::self.nth_point]
                            y_values = y_values[::self.nth_point]

                curve.setData(x_values, y_values)
        

        # redraw / update plot window
        QtWidgets.QApplication.processEvents()

    def plot_window_exit(self):
        """
        Waits for plot window to be closed.
        """
        time.sleep(0.5)
        if self.autoclose:
            self.win.close()
        else:
            print("Please close monitor window.") # TODO: print that into graph
            self.win.addLabel('You may close the window.', color='red', size='10pt')
            pg.QtWidgets.QApplication.exec_()

    def clear_temp_variables(self):
        """
        Clears any temporary variables created during the acquisition
        """
        for key in self.temp_variables:
            #print("clearing:", key)
            del self.__dict__[key]
        self.temp_variables = []

    def save_measurement(self, name, root='', save_channels='All', 
                         timestamp=True, comment=''):
        """Save acquired data.
        
        :param name: filename
        :param root: directory to save to
        :param save_channels: channel indices that are save. Defaults to 'All'.
        :param timestamp: include timestamp before 'filename'
        :param comment: commentary on the saved file
        """
        self.acquisition.save( name, root, save_channels, timestamp, comment)

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return np.array([freq, np.abs(amp)]).T

# TODO: write only 1 function for amplitude and phase, implement function arguments (*args)

def _fun_frf_amplitude(self, data):
    ch1, ch2 = data[-self.max_samples:].T
    
    # this code part is necessary for proper creation of new additional variables
    # used in this function:
    if not hasattr(self, 'var_H1'):
        self.var_freq = np.fft.rfftfreq(self.max_samples, d=1/self.acquisition.sample_rate)
        self.var_H1 = np.zeros(len(self.var_freq))
        self.var_n  = 0
        
        # these variables will be deleted from LDAQ class after acquisition run is stopped: 
        self.temp_variables.extend(["var_freq", "var_n", "var_H1" ]) 


    if len(ch1) == int(self.max_samples):
        ch1 = np.fft.rfft(ch1)
        ch2 = np.fft.rfft(ch2)
        Sxx = ch1 * np.conj(ch1)
        Sxy = ch2 * np.conj(ch1)
        H1 = Sxy / Sxx # frequency response function
       
         # FRF averaging:
        self.var_H1 = (self.var_H1 * self.var_n + H1) / (self.var_n + 1)
        self.var_n += 1

        return np.array([self.var_freq, np.abs(self.var_H1) ]).T
    
    else:
        return np.array([self.var_freq, np.abs(self.var_H1) ]).T

def _fun_frf_phase(self, data):
    ch1, ch2 = data[-self.max_samples:].T
    
    # this code part is necessary for proper creation of new additional variables
    # used in this function:
    if not hasattr(self, 'var_H1_2'):
        self.var_freq_2 = np.fft.rfftfreq(self.max_samples, d=1/self.acquisition.sample_rate)
        self.var_H1_2 = np.zeros(len(self.var_freq_2))
        self.var_n_2  = 0
        
        # these variables will be deleted from LDAQ class after acquisition run is stopped: 
        self.temp_variables.extend(["var_freq_2", "var_n_2", "var_H1_2" ]) 


    if len(ch1) == int(self.max_samples):
        ch1 = np.fft.rfft(ch1)
        ch2 = np.fft.rfft(ch2)
        Sxx = ch1 * np.conj(ch1)
        Sxy = ch2 * np.conj(ch1)
        H1 = Sxy / Sxx # frequency response function
       
         # FRF averaging:
        self.var_H1_2 = (self.var_H1_2 * self.var_n_2 + H1) / (self.var_n_2 + 1)
        self.var_n_2 += 1

        return np.array([self.var_freq_2, 180/np.pi* np.angle(self.var_H1_2) ]).T
    
    else:
        return np.array([self.var_freq_2, 180/np.pi* np.angle(self.var_H1_2) ]).T


# open measurements:
def load_measurement(name, directory='' ):
    with open(directory+'/' + name, 'rb') as f:
        return pickle.load(f)


def auto_nth_point(plot_layout, max_time, sample_rate, max_points_to_refresh=1e5):
    """
    Automatically determine the skip interval for drawing points.
    """
    plots = 0
    for key, val in plot_layout.items():
        if all(type(_) == int or type(_) == tuple for _ in val):
            plots += len(val)
        else:
            plots += 1
    
    points_to_refresh = max_time*sample_rate*plots
    
    if max_points_to_refresh < points_to_refresh:
        nth = int(np.ceil(points_to_refresh/max_points_to_refresh))
    else:
        nth = 1
    return nth
        
        
def identify_time_delay(measurement_dict, sourceA, sourceB, duration=10, freq_range=(1, 12)):
    """
    Identify the time delay between two sources by analyzing their cross-spectral density (CSD).

    Parameters:
    ldaq (object): An instance of the data acquisition class.
    sourceA (str): The key for the first source in the measurement dictionary.
    sourceB (str): The key for the second source in the measurement dictionary.
    duration (int, optional): Duration of the acquisition in seconds. Default is 10 seconds.
    freq_range (tuple, optional): A tuple specifying the frequency range [Hz] for analysis. Default is (1, 12).

    Returns:
    float: The estimated time delay between the two sources.
    """
    from scipy.interpolate import interp1d
    from scipy.signal import csd

    # Extract data and time for both sources
    dataA = measurement_dict[sourceA]['data']
    timeA = measurement_dict[sourceA]['time']
    dataB = measurement_dict[sourceB]['data']
    timeB = measurement_dict[sourceB]['time']

    # Interpolate data onto a common time vector
    t_int = np.linspace(0, min(timeA[-1], timeB[-1]), 100000 )
    fs = 1/(t_int[1]-t_int[0])
    v1_a, v2_a = interp1d(timeA, dataA.T)(t_int)
    v1_b, v2_b = interp1d(timeB, dataB.T)(t_int)

    # Calculate the cross-spectral density (CSD)
    nperseg = fs*4
    freq, _ = csd(v1_a, v1_b, fs=fs, nperseg=nperseg)
    sel = (freq < freq_range[1]) & (freq > freq_range[0])

    # Compute the transfer function
    H_v1 = csd(v1_a, v1_b, fs=fs, nperseg=nperseg)[1] / csd(v1_a, v1_a, fs=fs, nperseg=nperseg)[1]
    H_v1 = H_v1[sel]
    freq = freq[sel]

    # Calculate the phase and fit a linear function to the unwrapped phase
    phase = np.unwrap(np.angle(H_v1))
    k, n = np.polyfit(freq, phase, deg=1)

    # Calculate the time delay
    time_delay = k / (2 * np.pi)

    return time_delay
