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

        # self.maxTime = 5.0                                               # max time of data history on scrolling plots
        # self.max_samples = int(self.maxTime*self.acquisition.sample_rate) # max samples to display on some plots based on self.maxTime        
        # self.plot_channel_layout = "default"
        # self.nth_point = 50

        # self.plot_channel_layout = {
        #    (0, 0): [0],    (0, 1):[1],
        #    (1, 0): [0, 1], (1, 1):[0,1]
        # }

    def configure(self, plot_layout='default', max_time=5.0, nth_point=50, autoclose=False):
        """Configure the plot window settings.
        
        :param plot_layout: layout of the plots and channels. "default" or dict. Keys of dict are (axis 0, axis 1) of the subplot
            layout, values are lists of channel indices to show on the subplots.
        :param max_time: max time to show on the plot.
        :param nth_point: Only show every n_th point on the live plot."""
        self.plot_channel_layout = plot_layout
        self.maxTime = max_time
        self.nth_point = nth_point
        self.autoclose = autoclose
        
        self.max_samples = int(self.maxTime*self.acquisition.sample_rate) # max samples to display on some plots based on self.maxTime        

    def run(self):
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
        # time.sleep(0.1)

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

        return running

    def check_events(self):
        """
        Function that would disable DAQ, for example keyboard presses
        """
        if keyboard.is_pressed('q'):
            self.acquisition.stop()
            if self.generation != None:
                self.generation.stop()

    def plot_window_init(self):
        """
        Initializes plot window.
        """

        # Create app:
        # if not hasattr(self, "app"):
        #     self.app = QtGui.QApplication(sys.argv)   # initialize QT application if not yet initialized          

        self.win = pg.GraphicsLayoutWidget(show=True) # creates plot window
        self.win.setWindowTitle('Measurement Monitoring')
                 
        def create_plot(colspan=1, label_x="", label_y="", unit_x="", unit_y=""):
            # subplot options:
            p = self.win.addPlot(colspan=colspan)  
            p.setLabel('bottom', label_x, unit_x)
            p.setLabel('left', label_y, unit_y)
            p.setDownsampling(mode='peak')
            p.addLegend()
            p.setRange(xRange=[-self.maxTime, 0])
            p.setLimits(xMax=0)
            return p

        def create_curves(plot, channels):
            """
            channels - index
            """
            color_list = ["w", "g", "r", "b"]
            curves = []
            for i, channel in enumerate(channels):
                channel_name = self.acquisition.channel_names[channel]
                color = color_list[ i%len(color_list) ]
                curve = plot.plot( pen=pg.mkPen(color, width=1), name=channel_name) 
                curves.append(curve)
            return curves

        if self.plot_channel_layout == "default":
            self.plot_channel_layout = {(0, 0): np.arange( len(self.acquisition.channel_names) )}

        # create window layout:
        self.curves_dict = {}

        pos_x_curr = 0
        positions = list(self.plot_channel_layout.keys())
        positions = [positions[i] for i in np.lexsort(np.array(positions).T[::-1])]
        for pos_x, pos_y in positions:
            if pos_x != pos_x_curr:
                self.win.nextRow()

            plot = create_plot(colspan=1, label_x="", label_y="", unit_x="", unit_y="")
            channels = self.plot_channel_layout[ (pos_x, pos_y) ]
            curves = create_curves(plot, channels)
            self.curves_dict[(pos_x, pos_y)] = curves

            pos_x_curr = pos_x

            # initialize some data:
            dummy_data = np.zeros(self.max_samples ) # dummy data
            for curve in curves:
                curve.setData(dummy_data)
                curve.setPos(0, 0)

        QtGui.QApplication.processEvents()
        # time.sleep(0.1) # time to stabilize 
        #test

    def plot_window_update(self):
        """
        Updates plot window with collected data.
        """

        data = self.acquisition.plot_data[-self.max_samples:]
        # create fixed time array:
        self.time_arr = -1*(np.arange(data.shape[0])/self.acquisition.sample_rate)[::-self.nth_point]
        data = data[::self.nth_point]


        for position in self.plot_channel_layout:
            channels = self.plot_channel_layout[ position ]
            curves = self.curves_dict[ position ]

            for channel, curve in zip(channels, curves):
                x_values = self.time_arr
                y_values = data[:, channel]
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
            pg.QtGui.QApplication.exec_()


        
        
    