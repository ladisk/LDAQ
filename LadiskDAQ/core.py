import time
import os
import numpy as np
import matplotlib.pyplot as plt
import keyboard

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys

import threading


class LDAQ():
    def __init__(self, acquisition, generation=None):
        self.acquisition = acquisition
        self.generation  = generation

        # plot window settings:
        self.configure()

    def configure(self, plot_layout='default', max_time=5.0, nth_point=50, autoclose=False, fun="fft"):
        """Configure the plot window settings.
        
        :param plot_layout: layout of the plots and channels. "default" or dict. Keys of dict are (axis 0, axis 1) of the subplot
            layout, values are lists of channel indices to show on the subplots. See examples below.
        :param max_time: max time to show on the plot.
        :param nth_point: Only show every n_th point on the live plot.
        :param autoclose: Autoclose the window after the measurement is done.
        
        Plot layout
        -----------
        With plot layout, the user can define on which subplot the channels will be plotted. An example of
        plot layout is:
        >>> plot_layout = {
            (0, 0): [0, 1],
            (0, 1): [2, 3]
        }

        On first subplot (0, 0), channels 0 and 1 will be plotted, on the second subplot (0, 1), channels 2 and 3 will be plotted.
        Channels can also be plotted one against another. If, for example, we wish to plot channel 1 as a function of channel 0, input
        channel indices in a tuple; first the channel to be plotted on x-axis, and second the channel to be plotted on y-axis:
        >>> plot_layout = {
            (0, 0): [0, 1],
            (0, 1): [2, 3],
            (1, 0): [(0, 1)]
        }

        Additionally, the FFT of the signal can be computed on the fly. To define the subplots and channels where the FFT is computed, input the channel
        indices as a tuple (not as a list as was shown before):
        >>> plot_layout = {
            (0, 0): [0, 1], # Time series
            (0, 1): [2, 3], # Time series
            (1, 0): [(0, 1)], # 1 = f(0)
            (1, 1): (0,) # FFT(0)
        }
        
        """
        self.plot_channel_layout = plot_layout
        self.maxTime = max_time
        self.nth_point = nth_point
        self.autoclose = autoclose

        if type(fun) == str:
            self.fun_name = fun
            if fun == "fft":
                self.fun = _fun_fft
        else:
            self.fun_name = "custom"
            self.fun = fun
        
        self.max_samples = int(self.maxTime*self.acquisition.sample_rate) # max samples to display on some plots based on self.maxTime        

    def run(self):
        print('Press "q" to stop measurement.')
        print('\tWaiting for trigger...', end='')
        self.acquisition_started = False

        thread_list = []
        
        # Make separate threads for data acquisition
        thread_acquisition = threading.Thread(target=self.acquisition.run_acquisition )
        thread_list.append(thread_acquisition)

        # If generation is present, create generation thread
        if self.generation != None:
            thread_generation  = threading.Thread(target=self.generation.run_generation )
            thread_list.append(thread_generation)

        # initialize plot window:
        self.plot_window_init()

        # start both threads:
        for thread in thread_list:
            thread.start()
        time.sleep(0.1)

        # while data is being generated and collected:
        while self.is_running():
            self.check_events()

            # update plot window:
            self.plot_window_update()

        # after DAQ is completed, join the threads with the main thread:
        for thread in thread_list:
            thread.join()

        self.plot_window_exit() # waits for plot window to be closed.

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

            print('stop.')

        return running

    def check_events(self):
        """
        Function that would disable DAQ, for example keyboard presses
        """
        if keyboard.is_pressed('q'):
            self.acquisition.stop()
            if self.generation != None:
                self.generation.stop()

    def _create_plot(self, channels, pos_x, pos_y, label_x="", label_y="", unit_x="", unit_y=""):
        # subplot options:
        p = self.win.addPlot(row=pos_x, col=pos_y) 
        p.setLabel('bottom', label_x, unit_x)
        p.setLabel('left', label_y, unit_y)
        p.setDownsampling(mode='peak')
        p.addLegend()
        p.showGrid(x = True, y = True, alpha = 0.3)  

        if type(channels) == tuple:          # if fft
            if self.fun_name == "fft":
                p.setLogMode(x=None, y=True)
                p.setRange(xRange=[0, self.acquisition.sample_rate/4]) 
            
            pass
        elif type(channels[0]) == tuple:     # if channel vs. channel
            pass
        else:                                # if time signal
            p.setRange(xRange=[-self.maxTime, 0]) 

        curves = self._create_curves(p, channels)

        return p, curves

    def _create_curves(self, plot, channels):
        """
        channels - index
        """
        color_list = ["blue", "orange", "green", "red"]
        curves = []
        for i, channel in enumerate(channels):
            color = color_list[ i%len(color_list) ]

            if type(channel) == tuple:
                channel_name_x = self.acquisition.channel_names[channel[0]]
                channel_name_y = self.acquisition.channel_names[channel[1]]
                label = f"{channel_name_y} = f({channel_name_x})"
            else:
                channel_name_x = "Time"
                channel_name_y = self.acquisition.channel_names[channel]
                label = channel_name_y

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

        if self.plot_channel_layout == "default":
            self.plot_channel_layout = {(0, 0): np.arange( len(self.acquisition.channel_names) )}

        # create window layout:
        self.curves_dict = {}
        self.plot_dict = {}
        positions = list(self.plot_channel_layout.keys())
        positions = [positions[i] for i in np.lexsort(np.array(positions).T[::-1])]
        #ncols = len(set( [pos[1] for pos in positions ])  )
        #nrows = len(set( [pos[0] for pos in positions ])  )

        for i, (pos_x, pos_y) in enumerate(positions):

            # create subplot and curves on the subplot:
            channels = self.plot_channel_layout[(pos_x, pos_y)]
            plot, curves = self._create_plot(channels=channels, pos_x=pos_x, pos_y=pos_y, label_x="", label_y="", unit_x="", unit_y="")
            self.curves_dict[(pos_x, pos_y)] = curves
            self.plot_dict[(pos_x, pos_y)] = plot

    def plot_window_update(self):
        """
        Updates plot window with collected data.
        """

        data = self.acquisition.plot_data[-self.max_samples:]
        # create fixed time array:
        self.time_arr = -1*(np.arange(data.shape[0])/self.acquisition.sample_rate)
        data = data[::self.nth_point]

        if not self.acquisition_started and self.acquisition.Trigger.triggered:
            self.win.setBackground('lightgreen')
            print('triggered.')
            print('\tRecording...', end='')
            self.acquisition_started = True

        for position in self.plot_channel_layout:
            channels = self.plot_channel_layout[ position ]
            curves = self.curves_dict[ position ]

            for channel, curve in zip(channels, curves):
                if type(channel) == tuple:
                    channel_x, channel_y = channel
                    x_values = data[:, channel_x]
                    y_values = data[:, channel_y]
                elif type(channels) == list:
                    x_values = self.time_arr[::-self.nth_point]
                    y_values = data[:, channel]
                else:
                    fun_return = self.fun(self, self.acquisition.plot_data[-self.max_samples:][:, channel], position)
                    if type(fun_return) != tuple:
                        y_values = fun_return[::-self.nth_point]
                        x_values = self.time_arr[::-self.nth_point]
                    else:
                        x_values, y_values = fun_return
                
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

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data, position):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return freq, np.abs(amp)


        
        
    