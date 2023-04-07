import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

import numpy as np
import sys
import random
import time
import types

from .visualization_helpers import auto_nth_point, _fun_fft


INBUILT_FUNCTIONS = {'fft': _fun_fft}


class Visualization:
    def __init__(self, max_plot_time=1, layout=None, subplot_options=None, nth='auto', refresh_rate=100):
        """Live visualization of the measured data.
        
        :param max_plot_time: Maximum time that can be plotted.
        :param layout: Dictionary containing the source names and subplot layout with channel definitions.
            See examples below.
        :param subplot_options: Dictionary containing the options for each of the subplots (xlim, ylim, axis_style).
        :param nth: Number of points to skip when plotting. If 'auto', the number of points to skip is automatically determined.
        :param refresh_rate: Refresh rate of the plot in ms.

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


        Subplot options
        ---------------

        >>> subplot_options = {
            (0, 0): {
                'xlim': (0, 2),
                'ylim': (-1, 1),
                'axis_style': 'linear'
            },
            (0, 1): {
                'xlim': (0, 25),
                'ylim': (1e-5, 10),
                'axis_style': 'semilogy'
            },
            (1, 0): {
                'xlim': (0, 5),
                'axis_style': 'linear'
            },
            (1, 1): {
                'xlim': (0, 2),
                'axis_style': 'linear'
            }
        }

        For each subplot, e.g., (0, 0), the xlim and ylim can be set. Additionally, the axis style can be selected.
        Valid axis styles are:

        - linera (default)
        - semilogy
        - semilogx
        - loglog
        """
        self.layout = layout
        self.subplot_options = subplot_options
        self.max_plot_time = max_plot_time
        self.show_legend = True
        self.nth = nth
        self.refresh_rate = refresh_rate

        self.max_points_to_refresh = 1e5 # max number of points to refresh in all plots combined

        self.max_plot_time_per_subplot = {}
        if self.subplot_options is not None:
            for pos, options in self.subplot_options.items():
                if 'xlim' in options.keys():
                    self.max_plot_time_per_subplot[pos] = options['xlim'][1] - options['xlim'][0]

        
    def run(self, core):
        self.core = core

        if self.nth == 'auto':
            sample_rate = max([_.sample_rate for _ in self.core.acquisitions])
            self.nth = auto_nth_point(self.layout, self.max_plot_time, sample_rate, max_points_to_refresh=self.max_points_to_refresh)

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

        self.triggered = False

        self.layout = self.vis.layout
        self.setWindowTitle('Data Acquisition and Visualization')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_widget = QVBoxLayout(self.central_widget)
        self.layout_widget.setContentsMargins(20, 20, 20, 20) # set the padding

        self.add_buttons()

        self.init_plots()
        self.init_timer()

        self.showFullScreen()

    def add_buttons(self):
        self.button_layout = QHBoxLayout()

        self.close_button = QPushButton('Trigger')
        self.close_button.clicked.connect(self.trigger_measurement)
        self.button_layout.addWidget(self.close_button)

        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close_app)
        self.button_layout.addWidget(self.close_button)

        self.full_screen_button = QPushButton('Exit Full Screen')
        self.full_screen_button.clicked.connect(self.toggle_full_screen)
        self.button_layout.addWidget(self.full_screen_button)

        self.legend_button = QPushButton('Toggle Legends')
        self.legend_button.clicked.connect(self.toggle_legends)
        self.button_layout.addWidget(self.legend_button)

        self.layout_widget.addLayout(self.button_layout)

    
    def init_plots(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.time_start = time.time()
        self.plots = {}
        grid_layout = pg.GraphicsLayoutWidget()

        self.layout_widget.addWidget(grid_layout)
        self.subplots = {}
        self.legends = []

        color_dict = {}
        for source, positions in self.layout.items():
            channel_names = self.core.acquisitions[self.core.acquisition_names.index(source)].channel_names
            color_dict.update({ch: ind+len(color_dict) for ind, ch in enumerate(channel_names)})

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
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[y]], width=2), name=f"{channel_names[x]} vs. {channel_names[y]}")
                        plot_channels.append((line, pos, apply_function, x, y))
                    elif isinstance(ch, int):
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[ch]], width=2), name=f"{channel_names[ch]}")
                        plot_channels.append((line, pos, apply_function, ch))
                    elif isinstance(ch, types.FunctionType):
                        pass
                
                if self.vis.show_legend:
                    # Add legend to the subplot
                    legend = self.subplots[pos].addLegend()
                    for item in self.subplots[pos].items:
                        if isinstance(item, pg.PlotDataItem):
                            legend.addItem(item, item.opts['name'])
                    self.legends.append(legend)

            self.plots[source] = plot_channels


    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.vis.refresh_rate)


    def update_plots(self):
        if not self.core.is_running_global:
            self.close_app()

        if self.core.triggered_globally and not self.triggered:
            self.set_triggered_color()

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
                        y = fun_return[::self.vis.nth]
                        x = new_data[source]["time"][::self.vis.nth]
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


    def trigger_measurement(self):
        self.core.start_acquisition()


    def toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
            self.full_screen_button.setText('Full Screen')
        else:
            self.showFullScreen()
            self.full_screen_button.setText('Exit Full Screen')


    def toggle_legends(self):
        if self.vis.show_legend:
            self.vis.show_legend = False
        else:
            self.vis.show_legend = True

        for legend in self.legends:
            legend.setVisible(self.vis.show_legend)


    def set_triggered_color(self):
        self.triggered = True
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(152, 251, 152))
        self.setPalette(palette)


