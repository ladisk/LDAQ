import enum
from msilib.schema import Error
from re import X
import time
import os
from xml.sax import SAXException
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from beautifultable import BeautifulTable

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys

import threading
import types
import pickle

AXIS_SCALES = ['logx', 'logy']
INBUILT_FUNCTIONS = ['fft', 'frf_amp', 'frf_phase']


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

    def configure(self, plot_layout='default', max_time=5.0, nth_point='auto', autoclose=True, refresh_interval=0.01):
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
        self.plot_channel_layout = plot_layout
        if self.plot_channel_layout == "default":
            self.plot_channel_layout = {(0, 0):  [i for i in range(len(self.acquisition.channel_names)) ] }

        self.maxTime = max_time
        self.autoclose = autoclose
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
        self.plot_window_init()

        self.FREEZE_PLOT = False

        # start both threads:
        for thread in self.thread_list:
            thread.start()
        time.sleep(0.1)

        # while data is being generated and collected:
        while self.is_running():
            time.sleep(self.refresh_interval)

            self.check_events()

            # update plot window:
            if not self.FREEZE_PLOT:
                self.plot_window_update()


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

            self.plot_window_exit() # waits for plot window to be closed.
            self.clear_temp_variables()

        return running

    def check_events(self):
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
        QtGui.QApplication.processEvents()

    def plot_window_exit(self):
        """
        Waits for plot window to be closed.
        """
        if self.autoclose:
            self.win.close()
        else:
            print("Please close monitor window.") # TODO: print that into graph
            self.win.addLabel('You may close the window.', color='red', size='10pt')
            pg.QtGui.QApplication.exec_()

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
        
        
    