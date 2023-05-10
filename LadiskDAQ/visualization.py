import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton, QHBoxLayout, QDesktopWidget, QProgressBar, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen

import numpy as np
import sys
import random
import time
import types
import keyboard
from pyTrigger import RingBuffer2D

from .visualization_helpers import auto_nth_point, check_subplot_options_validity, _fun_fft


INBUILT_FUNCTIONS = {'fft': _fun_fft}


class Visualization:
    def __init__(self, layout=None, subplot_options=None, nth='auto', refresh_rate=100):
        """Live visualization of the measured data.

        For more details, see [documentation](https://ladiskdaq.readthedocs.io/en/latest/visualization.html).
        
        :param layout: Dictionary containing the source names and subplot layout with channel definitions.
            See examples below.
        :param subplot_options: Dictionary containing the options for each of the subplots (xlim, ylim, axis_style, etc.).
        :param nth: Number of points to skip when plotting. If 'auto', the number of points to skip is automatically determined
            in a way to make ``max_points_to_refresh = 1e4``. ``max_points_to_refresh`` is the attribute of the Visualization class and
            can be changed by the user.
        :param refresh_rate: Refresh rate of the plot in ms.

        The ``layout``
        --------------

        The layout of the live plot is set by the ``layout`` argument. An example of the ``layout`` argument is:

        >>> layout = {
                'DataSource': {
                    (0, 0): [0, 1],
                    (1, 0): [2, 3],
                }
            }

        This is a layout for a single acquisition source with name "DataSource". 
        When multiple sources are used, the name of the source is used as the key in the ``layout`` dictionary. 
        The value at each acquisition source is a dictionary where each key is a tuple of two integers. 
        The first integer is the row number and the second integer is the column number of the subplots.

        For the given example, the plot will have two subplots, each in one row.

        For each subplot, the data is then specified. 
        If the value is a list of integers, each integer corresponds to the index in the acquired data.
        For example, for the subplot defined with:

        >>> (0, 0): [0, 1]

        data with indices 0 and 1 will be plotted in the subplot at location (0, 0).

        Plotting from multiple sources
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        When plotting from multiple sources, the layout is defined:

        >>> layout = {
                'DataSource1': {
                    (0, 0): [0, 1],
                    (1, 0): [2, 3],
                },
                'DataSource2': {
                    (0, 0): [0],
                    (0, 1): [1]
                    (1, 1): [2, 3],
                }
            }

        Notice the different names of the sources. Each name corresponds to the name of the acquisition source, defined in the acquisition class.

        It is important to note that the subplot locations are the same for all acquisition sources, but the indices of the data are different. 

        For example, the subplot at location ``(0, 0)``
        will containt the plots from source "DataSource1" with indices 0 and 1, and the plots from source "DataSource2" with indices 0.

        Channel vs. channel plot
        ~~~~~~~~~~~~~~~~~~~~~~~~

        When plotting from multiple sources, it is possible to plot the data from one channel of one source against the data from one channel of another source.
        Example:

        >>> layout = {
                'DataSource': {
                    (0, 0): [0, 1],
                    (1, 0): [(2, 3)],
                }
            }

        In subplot at location (1, 0), the data from channel 3 will be plotted as a function of the data from channel 2.
        The first index of the ``tuple`` is considered the x-axis and the second index is considered the y-axis.

        The ``function`` option
        ~~~~~~~~~~~~~~~~~~~~~~~

        The data can be processed on-the-fly by a specified function.
        The functmion is added to the ``layout`` dictionary as follows:

        >>> layout = {
                'DataSource': {
                    (0, 0): [0, 1],
                    (1, 0): [2, 3, function],
                }
            }

        The ``function`` can be specified by the user. To use the built-in functions, a string is passed to the ``function`` argument. 
        An example of a built-in function is "fft" which computes the [Fast Fourier Transform](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
        of the data with indices 2 and 3.

        To build a custom function, the function must be defined as follows:

        >>> def function(self, channel_data):
                '''
                :param self: instance of the acquisition object (has to be there so the function is called properly)
                :param channel_data: channel data
                '''
                return channel_data**2

        The ``self`` argument in the custom function referes to the instance of the acquisition object. 
        This connection can be used to access the properties of the acquisition object, e.g. sample rate.
        The ``channel_data`` argument is a list of numpy arrays, where each array corresponds to the data from one channel. 
        The data is acquired in the order specified in the ``layout`` dictionary.

        For the layout example above, the custom function is called for each channel separetely, the ``channel_data`` is a one-dimensional numpy array. 
        To add mutiple channels to the ``channel_data`` argument,
        the ``layout`` dictionary is modified as follows:

        >>> layout = {
                'DataSource': {
                    (0, 0): [0, 1],
                    (1, 0): [(2, 3), function],
                }
            }

        The ``function`` is now passed the ``channel_data`` with shape (N, 2) where N is the number of samples.
        The function can also return a 2D numpy array with shape (N, 2) where the first column is the x-axis and the second column is the y-axis.
        An example of such a function is:

        >>> def function(self, channel_data):
                '''
                :param self: instance of the acquisition object (has to be there so the function is called properly)
                :param channel_data: 2D channel data array of size (N, 2)
                :return: 2D array np.array([x, y]).T that will be plotted on the subplot.
                '''
                ch0, ch1 = channel_data.T
                x =  np.arange(len(ch1)) / self.acquisition.sample_rate # time array
                y = ch1**2 + ch0 - 10
                return np.array([x, y]).T

        The ``subplot_options``
        -----------------------

        The properties of each subplot, defined in ``layout`` can be specified with the ``subplot_options`` argument. The ``subplot_options`` argument is a dictionary where the keys are the positions of the subplots.

        Example:

        >>> subplot_options = {
                (0, 0): {
                    'xlim': (0, 2),
                    'ylim': (-5, 5),
                    'axis_style': 'linear',
                    'title': 'My title 1'
                },
                (0, 1): {
                    'xlim': (0, 25),
                    'ylim': (1e-5, 1e3),
                    'axis_style': 'semilogy',
                    'title': 'My title 2'
                },
                (1, 0): {
                    'xlim': (-5, 5),
                    'ylim': (-5, 5),
                    'axis_style': 'linear',
                    'title': 'My title 3'
                },
                (1, 1): {
                    'xlim': (0, 2),
                    'axis_style': 'linear',
                    'title': 'My title 4'
                }
            }

        Currently, the following options are available:

        - ``xlim``: tuple of two floats, the limits of the x-axis.
        - ``ylim``: tuple of two floats, the limits of the y-axis.
        - ``t_span``: int/float, the length of the time axis. If this option is not specified, it is computed from the ``xlim``.
        - ``axis_style``: string, the style of the axis. Can be "linear", "semilogx", "semilogy" or "loglog".
        - ``title``: string, the title of the subplot.
        - ``rowspan``: int, the number of rows the subplot spans. Default is 1.
        - ``colspan``: int, the number of columns the subplot spans. Default is 1.
        - ``refresh_rate``: int, the refresh rate of the subplot in milliseconds. 
        If this option is not specified, the refresh rate defined in the :class:`Visualization` is used.
        - ``nth``: int, same as the ``nth`` argument in :class:`Visualization`. 
        If this option is not specified, the ``nth`` argument defined in the :class:`Visualization` is used.

        .. note:: 
            When plotting a simple time signal, the ``t_span`` and ``xlim`` have the same effect. 
            
            However, when plotting channel vs. channel, the ``t_span`` specifies the time range of the data and the ``xlim`` specifies the range of the x-axis (spatial).

            When plotting a function, the ``t_span`` determines the time range of the data that is passed to the function. 
            Last ``t_span`` seconds of data are passed to the function.


        .. note::
            The ``xlim`` defines the samples that are plotted on the x-axis, not only a narrowed view of the data. 
            With this, the same data can be viewed with different zoom levels in an effcient way.

        An example of ``subplot_options`` with ``colspan``:

        >>> subplot_options = {
                (0, 0): {
                    'xlim': (0, 2),
                    'ylim': (-5, 5),
                    'axis_style': 'linear',
                    'title': 'My title 1',
                    'colspan': 2,
                },
                (1, 0): {
                    'xlim': (-5, 5),
                    'ylim': (-5, 5),
                    'axis_style': 'linear',
                    'title': 'My title 3'
                },
                (1, 1): {
                    'xlim': (0, 2),
                    'axis_style': 'linear',
                    'title': 'My title 4',
                    'rowspan': 2
                },
            }

        Note that the subplot at location (0, 1) must be omitted, since it is spanned by the subplot at location (0, 0).
        The subplot at location (0, 1) must also be omitted in the ``layout``.
        """
        self.layout = layout
        self.subplot_options = subplot_options
        self.max_plot_time = 1
        self.show_legend = True
        self.nth = nth
        self.refresh_rate = refresh_rate
        
        self.update_refresh_rate = 10 # [ms] interval of calling the plot_update function
        self.max_points_to_refresh = 1e4

        # check validity of the layout (all keys must be tuples or all keys must be strings)
        if self.layout is not None:
            if any(isinstance(k, tuple) for k in self.layout.keys()) and not all(isinstance(k, tuple) for k in self.layout.keys()):
                raise ValueError("Invalid layout.")

        # check validity of the nth parameter:
        if self.nth == 'auto':
            if self.subplot_options is None or self.layout is None:
                print('Warning: `nth` could not be determined automatically. Using `nth=1`.')
                self.nth = 1
        elif isinstance(self.nth, int):
            pass
        else:
            raise ValueError('`nth` must be an integer or "auto".')
        
        # check validity of the subplot_options (rowspan/colspan)
        if self.subplot_options is not None:
            if not check_subplot_options_validity(self.subplot_options):
                raise ValueError("Invalid subplot options. Check the `rowspan` and `colspan` values.")
    

    def run(self, core):
        self.core = core
        # self.core.is_running_global = False

        # Check if the layout is valid.
        self.check_layout()

        # Check if the subplot options are valid.
        self.check_subplot_options()

        # Compute the nth point for each subplot.
        if not isinstance(self.nth, dict):
            self.nth = auto_nth_point(self.layout, self.subplot_options, self.core, max_points_to_refresh=self.max_points_to_refresh, known_nth=self.nth)

        # Create the ring buffers for each acquisition.
        self.create_ring_buffers()

        # Start the QT application.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        with self.app:
            self.main_window = MainWindow(self, self.core, self.app)
            self.main_window.show()
            self.app.exec_()
        
        
    def check_layout(self):
        if self.layout is None:
            # Make default layout.
            self.layout = {}
            for source in self.core.acquisition_names:
                acq = self.core.acquisitions[self.core.acquisition_names.index(source)]
                self.layout[source] = {(0, 0): list(range(acq.n_channels))}
        else:
            if all(isinstance(k, tuple) for k in self.layout.keys()):
                # If all keys are tuples, then the layout is for a single acquisition.
                self.layout = {self.core.acquisition_names[0]: self.layout}
            else:
                # If not, then the layout is for multiple acquisitions.
                pass


    def check_subplot_options(self):
        if self.subplot_options is None:
            # Make default subplot options.
            self.subplot_options = {}
            for source in self.core.acquisition_names:
                for pos in self.layout[source].keys():
                    self.subplot_options[pos] = {"xlim": (0, 1), "axis_style": "linear"}

        for pos, options in self.subplot_options.items():
            # Check that all subplots have `t_span` and `xlim` defined.
            if 'xlim' in options.keys() and 't_span' not in options.keys():
                self.subplot_options[pos]['t_span'] = options['xlim'][1] - options['xlim'][0]
            elif 't_span' in options.keys() and 'xlim' not in options.keys():
                self.subplot_options[pos]['xlim'] = (0, options['t_span'])
            elif 'xlim' not in options.keys() and 't_span' not in options.keys():
                self.subplot_options[pos]['xlim'] = (0, 1)
                self.subplot_options[pos]['t_span'] = 1
            else:
                pass

            # Define the refresh rate for each subplot.
            if 'refresh_rate' in options.keys():
                self.subplot_options[pos]['subplot_refresh_rate'] = self.update_refresh_rate*(options['refresh_rate']//self.update_refresh_rate)
            else:
                self.subplot_options[pos]['subplot_refresh_rate'] = self.update_refresh_rate*(self.refresh_rate//self.update_refresh_rate)


    def create_ring_buffers(self):
        self.ring_buffers = {}
        for source in self.layout.keys():
            acq = self.core.acquisitions[self.core.acquisition_names.index(source)]
            rows = int(max([self.subplot_options[pos]['t_span'] * acq.sample_rate for pos in self.layout[source].keys()]))
            self.ring_buffers[source] = RingBuffer2D(rows, acq.n_channels)


class MainWindow(QMainWindow):
    def __init__(self, vis, core, app):
        super().__init__()
        
        self.vis = vis
        self.core = core
        self.app = app

        self.triggered = False
        self.measurement_stopped = False
        self.freeze_plot = False

        self.layout = self.vis.layout
        self.setWindowTitle('Data Acquisition and Visualization')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_widget = QHBoxLayout(self.central_widget)
        self.layout_widget.setContentsMargins(20, 20, 20, 20) # set the padding

        self.desktop = QDesktopWidget().screenGeometry()
        if hasattr(self.vis, 'last_position'):
            self.move(self.vis.last_position)
            self.resize(self.vis.last_size)
        else:
            self.resize(int(self.desktop.width()*0.95), int(self.desktop.height()*0.8))

            window_geometry = self.frameGeometry()
            center_offset = self.desktop.center() - window_geometry.center()
            self.move(self.pos() + center_offset)

        self.add_buttons()

        self.init_plots()

        self.init_timer()


    def add_buttons(self):
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(5, 5, int(self.desktop.width()*0.01), 5)

        self.trigger_button = QPushButton('Start Measurement')
        self.trigger_button.clicked.connect(self.trigger_measurement)
        self.button_layout.addWidget(self.trigger_button)

        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close_app)
        self.button_layout.addWidget(self.close_button)

        self.full_screen_button = QPushButton('Full Screen')
        self.full_screen_button.clicked.connect(self.toggle_full_screen)
        self.button_layout.addWidget(self.full_screen_button)

        self.legend_button = QPushButton('Toggle Legends')
        self.legend_button.clicked.connect(self.toggle_legends)
        self.button_layout.addWidget(self.legend_button)

        self.freeze_button = QPushButton('Freeze')
        self.freeze_button.clicked.connect(self.toggle_freeze_plot)
        self.button_layout.addWidget(self.freeze_button)

        label = QLabel(self)
        label.setText("Measurement status:")
        self.button_layout.addWidget(label)

        self.label = QLabel(self)
        self.label.setText("Not started.")
        self.button_layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setOrientation(Qt.Vertical)

        self.progress_bar.setStyleSheet("""
            QProgressBar {
                width: 100px;
                height: 500px;
                padding: 0px;
                align: center;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
            }
        """)

        self.button_layout.addStretch(1)

        self.button_layout.addWidget(self.progress_bar)

        self.layout_widget.addLayout(self.button_layout)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            if self.measurement_stopped:
                self.close_app()
            else:
                self.stop_measurement(mode='manual')

        elif event.key() == Qt.Key_S:
            self.core.start_acquisition()
        
        elif event.key() == Qt.Key_F:
            self.toggle_freeze_plot()
        
        elif event.key() == Qt.Key_L:
            self.toggle_legends()

        elif event.key() == Qt.Key_F11:
            self.toggle_full_screen()

    
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
                    if 'rowspan' in self.vis.subplot_options[pos].keys():
                        rowspan = self.vis.subplot_options[pos]['rowspan']
                    else:
                        rowspan = 1
                    
                    if 'colspan' in self.vis.subplot_options[pos].keys():
                        colspan = self.vis.subplot_options[pos]['colspan']
                    else:
                        colspan = 1

                    if 'title' in self.vis.subplot_options[pos].keys():
                        title = self.vis.subplot_options[pos]['title']
                    else:
                        title = None

                    self.subplots[pos] = grid_layout.addPlot(*pos, rowspan=rowspan, colspan=colspan, title=title)

                    if self.vis.subplot_options is not None and pos in self.vis.subplot_options:
                        options = self.vis.subplot_options[pos]
                        transform_lim_x = lambda x: x
                        transform_lim_y = lambda x: x
                        if 'axis_style' in options:
                            if options['axis_style'] == 'semilogy':
                                self.subplots[pos].setLogMode(y=True)
                                transform_lim_y = lambda x: np.log10(x)
                            elif options['axis_style'] == 'semilogx':
                                self.subplots[pos].setLogMode(x=True)
                                transform_lim_x = lambda x: np.log10(x)
                            elif options['axis_style'] == 'loglog':
                                self.subplots[pos].setLogMode(x=True, y=True)
                            elif options['axis_style'] == 'linear':
                                self.subplots[pos].setLogMode(y=False)

                        if 'xlim' in options:
                            self.subplots[pos].setXRange(transform_lim_x(options['xlim'][0]), transform_lim_x(options['xlim'][1]))
                        if 'ylim' in options:
                            self.subplots[pos].setYRange(transform_lim_y(options['ylim'][0]), transform_lim_y(options['ylim'][1]))
                
                apply_function = lambda vis, x: x
                for ch in channels:
                    if isinstance(ch, types.FunctionType):
                        apply_function = ch
                    elif ch in INBUILT_FUNCTIONS.keys():
                        apply_function = INBUILT_FUNCTIONS[ch]
                    
                for ch in channels:
                    if not isinstance(ch, types.FunctionType) and ch not in INBUILT_FUNCTIONS.keys():
                        plot_channels.append({
                            'pos': pos,
                            'apply_function': apply_function,
                            'since_refresh': 1e40, # a very large number to ensure that the first refresh will always be done
                        })

                    if isinstance(ch, tuple):
                        x, y = ch
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[y]], width=2), name=f"{channel_names[x]} vs. {channel_names[y]}", clear=True)
                        plot_channels[-1]['line'] = line
                        plot_channels[-1]['channels'] = (x, y)
                        plot_channels[-1]['nth'] = self.vis.nth[source][pos][ch[0]]

                    elif isinstance(ch, int):
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[ch]], width=2), name=f"{channel_names[ch]}")
                        plot_channels[-1]['line'] = line
                        plot_channels[-1]['channels'] = (ch,)
                        plot_channels[-1]['nth'] = self.vis.nth[source][pos][ch]

                
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
        self.timer.start(self.vis.update_refresh_rate)


    def update_ring_buffers(self):
        new_data = self.core.get_measurement_dict_PLOT()
        for source, buffer in self.vis.ring_buffers.items():
            buffer.extend(new_data[source])


    def update_plots(self, force_refresh=False):
        # Stop the measurement if the acquisitions are done and if the measurement has not been stopped.
        if not self.core.is_running_global and not self.measurement_stopped:
            self.stop_measurement()

        # If the emasurement is started, start the timer and update the progress bar.
        if self.core.triggered_globally and not self.triggered:
            self.on_measurement_start()

            self.time_start = time.time()
            self.progress_bar.setMaximum(int(1000*self.core.acquisitions[0].trigger_settings['duration']))

        # If the measurement is running, update the progress bar and the label.
        if self.triggered and self.core.is_running_global:
            self.progress_bar.setValue(int(1000*(time.time() - self.time_start)))
            self.label.setText(f"{time.time() - self.time_start:.1f}/{self.core.acquisitions[0].trigger_settings['duration']:.1f} s")

        # Update the ring buffers.
        self.update_ring_buffers()

        if not self.freeze_plot:
            for source, plot_channels in self.plots.items():
                self.vis.acquisition = self.core.acquisitions[self.core.acquisition_names.index(source)]

                # for line, pos, apply_function, *channels in plot_channels:
                for plot_channel in plot_channels:
                    refresh_rate = self.vis.subplot_options[plot_channel['pos']]['subplot_refresh_rate']
                    since_refresh = plot_channel['since_refresh']

                    if refresh_rate <= since_refresh + self.vis.update_refresh_rate or force_refresh:
                        # If time to refresh, refresh the plot and set since_refresh to 0.
                        plot_channel['since_refresh'] = 0
                        
                        new_data = self.vis.ring_buffers[source].get_data()
                        self.update_line(new_data, plot_channel)
                    else:
                        # If not time to refresh, increase since_refresh by update_refresh_rate.
                        plot_channel['since_refresh'] += self.vis.update_refresh_rate
    

    def update_line(self, new_data, plot_channel):
        # only plot data that are within xlim (applies only for normal plot, not ch vs. ch)
        t_span_samples = int(self.vis.subplot_options[plot_channel['pos']]['t_span'] * self.vis.acquisition.sample_rate)
        
        nth = plot_channel['nth']

        xlim = self.vis.subplot_options[plot_channel['pos']]['xlim']

        if len(plot_channel['channels']) == 1: 
            # plot a single channel
            ch = plot_channel['channels'][0]
            fun_return = plot_channel['apply_function'](self.vis, new_data[-t_span_samples:, ch])

            if len(fun_return.shape) == 1: 
                # if function returns only 1D array
                y = fun_return[::nth]
                x = (np.arange(t_span_samples) / self.vis.acquisition.sample_rate)[::nth]

            elif len(fun_return.shape) == 2 and fun_return.shape[1] == 2:  
                # function returns 2D array (e.g. fft returns freq and amplitude)
                # In this case, the first column is the x-axis and the second column is the y-axis.
                # The nth argument is not used in this case.
                x, y = fun_return.T # expects 2D array to be returned

            else:
                raise Exception("Function used in `layout` must return either 1D array or 2D array with 2 columns.")
            
            mask = (x >= xlim[0]) & (x <= xlim[1]) # Remove data outside of xlim
            
            plot_channel['line'].setData(x[mask], y[mask])

        elif len(plot_channel['channels']) == 2: 
            # channel vs. channel
            fun_return = plot_channel['apply_function'](self.vis, new_data[-t_span_samples:, plot_channel['channels']])
            x, y = fun_return.T
            mask = (x >= xlim[0]) & (x <= xlim[1]) # Remove data outside of xlim
            
            plot_channel['line'].setData(x[mask][::nth], y[mask][::nth])
        
        else:
            raise Exception("A single channel or channel vs. channel plot can be plotted at a time. Got more than 2 channels.")


    def close_app(self):
        self.vis.last_position = self.pos()
        self.vis.last_size = self.size()

        if not self.measurement_stopped:
            self.stop_measurement()

        self.app.quit()
        self.close()

    
    def closeEvent(self, a0):
        """Call close_app() when the user closes the window by pressing the X button."""
        self.close_app()
        return super().closeEvent(a0)


    def stop_measurement(self, mode='finished'):
        self.core.triggered_globally = True # dummy start measurement
        self.core.stop_acquisition_and_generation()
        self.timer.stop()

        self.trigger_button.setText('Start measurement')
        self.trigger_button.setEnabled(False)
        self.measurement_stopped = True

        # Update the plots one last time.
        self.update_plots(force_refresh=True)

        # palette = self.palette()
        # palette.setColor(self.backgroundRole(), QColor(152, 251, 251))
        # self.setPalette(palette)

        if mode == 'finished':
            self.label.setText(f"Finished.")
            self.progress_bar.setValue(self.progress_bar.maximum())

        if self.core.autoclose:
            self.close_app()


    def trigger_measurement(self):
        if not self.triggered:
            self.core.start_acquisition()
        else:
            self.stop_measurement(mode='manual')
            

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


    def on_measurement_start(self):
        self.triggered = True
        self.trigger_button.setText('Stop measurement')
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(152, 251, 177))
        self.setPalette(palette)


    def toggle_freeze_plot(self):
        if self.freeze_plot:
            self.freeze_plot = False
            self.freeze_button.setText('Freeze')
        else:
            self.freeze_plot = True
            self.freeze_button.setText('Unfreeze')


        