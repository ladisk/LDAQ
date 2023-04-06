import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import sys
import random
import time
import types

from visualization_helpers import _fun_fft


INBUILT_FUNCTIONS = {'fft': _fun_fft}


class Visualization:
    def __init__(self, max_plot_time=1, layout=None, subplot_options=None, show_legend=True):
        self.layout = layout
        self.subplot_options = subplot_options
        self.max_plot_time = max_plot_time
        self.show_legend = show_legend

        self.max_plot_time_per_subplot = {}
        if self.subplot_options is not None:
            for pos, options in self.subplot_options.items():
                if 'xlim' in options.keys():
                    self.max_plot_time_per_subplot[pos] = options['xlim'][1] - options['xlim'][0]

        
    def run(self, core):
        self.core = core

        if self.layout is None:
            self.layout = {}
            for source in self.core.acquisition_names:
                acq = self.core.acquisitions[self.core.acquisition_names.index(source)]
                self.layout[source] = {(0, 0): list(range(acq.n_channels))}

        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        with self.app:
            self.main_window = MainWindow(self, self.core, self.app)
            self.main_window.show()
            self.app.exec_()
        
        self.core.is_running_global = False

class MainWindow(QMainWindow):
    def __init__(self, vis, core, app):
        super().__init__()
        
        self.vis = vis
        self.core = core
        self.app = app

        self.layout = self.vis.layout
        self.setWindowTitle('Data Acquisition and Visualization')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_widget = QVBoxLayout(self.central_widget)

        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close_app)
        self.layout_widget.addWidget(self.close_button)

        self.init_plots()
        self.init_timer()

    
    def init_plots(self):
        self.time_start = time.time()
        self.plots = {}
        grid_layout = pg.GraphicsLayoutWidget()
        self.layout_widget.addWidget(grid_layout)
        self.subplots = {}

        for source, positions in self.layout.items():
            channel_names = self.core.acquisitions[self.core.acquisition_names.index(source)].channel_names
            plot_channels = []
            for pos, channels in positions.items():
                if pos not in self.subplots.keys():
                    self.subplots[pos] = grid_layout.addPlot(*pos)
                    
                    if self.vis.subplot_options is not None and pos in self.vis.subplot_options:
                        options = self.vis.subplot_options[pos]
                        if 'axis_style' in options:
                            if options['axis_style'] == 'semilogy':
                                self.subplots[pos].setLogMode(y=True)
                            elif options['axis_style'] == 'semilogx':
                                self.subplots[pos].setLogMode(x=True)
                            elif options['axis_style'] == 'loglog':
                                self.subplots[pos].setLogMode(x=True, y=True)
                            elif options['axis_style'] == 'linear':
                                self.subplots[pos].setLogMode(y=False)

                        if 'xlim' in options:
                            self.subplots[pos].setXRange(*options['xlim'])
                        if 'ylim' in options:
                            self.subplots[pos].setYRange(*options['ylim'])
                
                apply_function = lambda vis, x: x
                for ch in channels:
                    if isinstance(ch, types.FunctionType):
                        apply_function = ch
                    elif ch in INBUILT_FUNCTIONS.keys():
                        apply_function = INBUILT_FUNCTIONS[ch]
                    
                for ch in channels:
                    if isinstance(ch, tuple):
                        x, y = ch
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()), name=f"{channel_names[x]} vs. {channel_names[y]}")
                        plot_channels.append((line, pos, apply_function, x, y))
                    elif isinstance(ch, int):
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()), name=f"{channel_names[ch]}")
                        plot_channels.append((line, pos, apply_function, ch))
                    elif isinstance(ch, types.FunctionType):
                        pass
                
                if self.vis.show_legend:
                    # Add legend to the subplot
                    legend = self.subplots[pos].addLegend()
                    for item in self.subplots[pos].items:
                        if isinstance(item, pg.PlotDataItem):
                            legend.addItem(item, item.opts['name'])

            self.plots[source] = plot_channels


    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)


    def update_plots(self):
        if not self.core.is_running_global:
            self.close_app()

        new_data = self.core.get_measurement_dict(self.vis.max_plot_time)
        for source, plot_channels in self.plots.items():
            self.vis.acquisition = self.core.acquisitions[self.core.acquisition_names.index(source)]

            for line, pos, apply_function, *channels in plot_channels:
                # only plot data that are within xlim (applies only for normal plot, not ch vs. ch)
                max_plot_samples =  int(self.vis.max_plot_time_per_subplot[pos] * self.vis.acquisition.sample_rate)

                if len(channels) == 1: # plot a single channel
                    ch = channels[0]
                    fun_return = apply_function(self.vis, new_data[source]["data"][:, ch])
                    if len(fun_return.shape) == 1: # if function returns only 1D array
                        y = fun_return
                        x = new_data[source]["time"]
                    else:  # function returns 2D array (e.g. fft returns freq and amplitude)
                        x, y = fun_return.T # expects 2D array to be returned

                    line.setData(x[:max_plot_samples], y[-max_plot_samples:])

                else: # channel vs. channel
                    channel_x, channel_y = channels
                    fun_return = apply_function(self.vis, new_data[source]['data'][:, [channel_x, channel_y]])
                    x, y = fun_return.T
                    line.setData(x, y)



    def close_app(self):
        self.timer.stop()
        self.app.quit()
        self.close()

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))



