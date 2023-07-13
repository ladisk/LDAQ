import pyqtgraph as pg
from pyqtgraph import ImageView, ImageItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton, QHBoxLayout, QDesktopWidget, QProgressBar, QLabel
from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QIcon

import numpy as np
import sys
import os
import random
import time
import types
import keyboard
from pyTrigger import RingBuffer2D

from typing import Optional, Tuple, Union, List, Callable

from .visualization_helpers import compute_nth, check_subplot_options_validity, _fun_fft, _fun_frf_amp, _fun_frf_phase, _fun_coh

INBUILT_FUNCTIONS = {'fft': _fun_fft, 'frf_amp': _fun_frf_amp, 'frf_phase': _fun_frf_phase, 'coh': _fun_coh}

# Create a subclass of ImageView
class HoverImageView(pg.ImageView):
    def __init__(self):
        super().__init__()

        # Create a label to display the pixel value
        self.pixel_label = QLabel(self)
        self.pixel_label.setStyleSheet("background-color: white; padding: 2px;")

        # Position the label on top of the ImageView
        self.pixel_label.setAlignment(Qt.AlignCenter)
        self.pixel_label.setFixedWidth(500)
        self.pixel_label.move(10, 10)

        # Save the last hover position
        self.last_hover_pos = None

        # Enable mouse tracking to receive hover events
        self.getImageItem().hoverEvent = self.hoverEvent
        self.setMouseTracking(True)

        # Create a timer to periodically check the pixel value under the cursor
        self.timer = QTimer()
        self.timer.timeout.connect(self.checkPixelValue)
        self.timer.start(100)  # check every 100 ms

    def hoverEvent(self, event):
        """Override the hover event handler. Record the last hover position, if the
        mouse is outside the image, set the last hover position to None."""
        if event.isExit():
            self.last_hover_pos = None
        else:
            self.last_hover_pos = event.pos()

    def checkPixelValue(self):
        """Because the hover event is only triggered when the mouse moves, this event is called within
        a timer to periodically check the pixel value under the cursor."""
        if self.last_hover_pos is not None:
            pos = self.last_hover_pos
            x = int(pos.x())
            y = int(pos.y())
            mapped_pos = self.view.mapFromItem(self.getImageItem(), QPointF(pos))
            x_m = int(mapped_pos.x())
            y_m = int(mapped_pos.y())
            image = self.getImageItem().image
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                value = image[y, x]
                # mapped_pos = self.view.mapFromItem(self.getImageItem(), QPointF(pos))
                self.pixel_label.setText(f'x:{x}, y:{y}, {value}')
                self.pixel_label.move(x_m + 10, y_m + 10)
            else:
                self.pixel_label.setText('')
        else:
            self.pixel_label.setText('')
            self.pixel_label.move(10, 10)

class Visualization:
    def __init__(self, refresh_rate: int = 100, max_points_to_refresh: int = 10000, sequential_plot_updates: bool = False):
        """Initialize a new `Visualization` object.

        Args:
            refresh_rate (int, optional): The refresh rate of the plot in milliseconds. Defaults to 100.
            max_points_to_refresh (int, optional): The maximum number of points to refresh in the plot. Adjust this number to optimize performance.
                This number is used to compute the `nth` value automatically. Defaults to 10000.
            sequential_plot_updates (bool, optional): If `True`, the plot is updated sequentially (one line at a time).
                If `False`, all lines are updated in each iteration of the main loop. Defaults to `True`.

        """
        self.max_plot_time = 1
        self.show_legend = True
        self.refresh_rate = refresh_rate
        self.plots = None
        self.subplot_options = {}
        self.add_line_widget = False
        self.add_image_widget = False
        
        self.update_refresh_rate = 10 # [ms] interval of calling the plot_update function
        self.max_points_to_refresh = max_points_to_refresh
        self.sequential_plot_updates = sequential_plot_updates
    

    def add_lines(self, position: Tuple[int, int], source: str, channels: Union[int, str, tuple, list],
                  function: Union[callable, str, None] = None, nth: Union[int, str] = "auto",
                  refresh_rate: Union[int, None] = None, t_span: Union[int, float, None] = None) -> None:
        """Build the layout dictionary.

        Args:
            position (tuple): The position of the subplot. Example: ``(0, 0)``.
            source (str): The source of the data. Name that was given to the ``Acquisition`` object.
            channels (int/str/tuple/list): The channels from the ``source`` to be plotted. Can also be a list of tuples of integers to plot channel vs. channel.
                Example: ``[(0, 1), (2, 3)]``. For more details, see example below and documentation.
            function (function/str, optional): The function to be applied to the data before plotting. If ``channels`` is a list of tuples,
                the function is applied to each tuple separately. Defaults to ``None``.
            nth (int/str, optional): The nth sample to be plotted. If ``nth`` is ``"auto"``, the nth sample is computed automatically.
                Defaults to ``"auto"``.
            refresh_rate (int, optional): The refresh rate of the subplot in milliseconds. If this argument is not specified, the
                refresh rate defined in the :class:`Visualization` is used. Defaults to ``None``.
            t_span (int/float, optional): The length of the time axis. If this option is not specified, it is computed from the ``xlim``.
                Defaults to ``None``.


        Channels
        ~~~~~~~~

        If the ``channels`` argument is an integer, the data from the channel with the specified index will be plotted.

        If the ``channels`` argument is a list of integers, the data from the channels with the specified indices will be plotted:

        >>> vis.add_lines(position=(0, 0), source='DataSource', channels=[0, 1])

        To plot channel vs. channel the ``channels`` argument is a tuple of two integers:

        >>> vis.add_lines(position=(0, 0), source='DataSource', channels=(0, 1))

        The first integer is the index of the x-axis and the second integer is the index of the y-axis.

        Multiple channel vs. channel plots can be added to the same subplot:

        >>> vis.add_lines(position=(0, 0), source='DataSource', channels=[(0, 1), (2, 3)])

        The ``function`` argument
        ~~~~~~~~~~~~~~~~~~~~~~~~~

        The data can be processed on-the-fly by a specified function.


        The ``function`` can be specified by the user. To use the built-in functions, a string is passed to the ``function`` argument. 
        An example of a built-in function is "fft" which computes the [Fast Fourier Transform](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
        of the data with indices 0 and 1:

        >>> vis.add_lines(position=(0, 0), source='DataSource', channels=[0, 1], function='fft')

        To build a custom function, the function must be defined as follows:

        >>> def function(self, channel_data):
                '''
                Args:
                    self: instance of the acquisition object (has to be there so the function is called properly)
                    channel_data (dict): A dictionary containing the channel data.
                '''
                return channel_data**2

        The ``self`` argument in the custom function referes to the instance of the acquisition object. 
        This connection can be used to access the properties of the acquisition object, e.g. sample rate.
        The ``channel_data`` argument is a list of numpy arrays, where each array corresponds to the data from one channel. 
        The data is acquired in the order specified in the ``channels`` argument.

        For the example above, the custom function is called for each channel separetely, the ``channel_data`` is a one-dimensional numpy array. 
        To add mutiple channels to the ``channel_data`` argument, the ``channels`` argument is modified as follows:

        >>> vis.add_lines(position=(0, 0), source='DataSource', channels=[(0, 1)], function=function)

        The ``function`` is now passed the ``channel_data`` with shape ``(N, 2)`` where ``N`` is the number of samples.
        The function can also return a 2D numpy array with shape ``(N, 2)`` where the first column is the x-axis and the second column is the y-axis.
        An example of such a function is:

        >>> def function(self, channel_data):
                '''
                Args:
                    self: instance of the acquisition object (has to be there so the function is called properly)
                    channel_data (np.ndarray): A 2D channel data array of size (N, 2).

                Returns:
                    np.ndarray: A 2D array np.array([x, y]).T that will be plotted on the subplot.
                '''
                ch0, ch1 = channel_data.T
                x =  np.arange(len(ch1)) / self.acquisition.sample_rate # time array
                y = ch1**2 + ch0 - 10
                return np.array([x, y]).T

        """
        self.add_line_widget = True

        if not isinstance(source, str):
            raise ValueError("The source must be a string.")
        if not isinstance(position, tuple):
            raise ValueError("The position must be a tuple.")
        if not (isinstance(channels, list) or isinstance(channels, tuple) or isinstance(channels, int)):
            raise ValueError("The channels must be a list, tuple or an integer.")
        if not (isinstance(function, types.FunctionType) or function in INBUILT_FUNCTIONS.keys() or function is None):
            raise ValueError("The function must be a function or a string.")
        if not (isinstance(nth, int) or nth == 'auto'):
            raise ValueError("The nth must be an integer or 'auto'.")

        if self.plots is None:
            self.plots = {}
        
        if source not in self.plots.keys():
            self.plots[source] = []

        if isinstance(channels, int) or isinstance(channels, tuple):
            channels = [channels]

        if isinstance(function, types.FunctionType):
            apply_function = function
        elif function in INBUILT_FUNCTIONS.keys():
            apply_function = INBUILT_FUNCTIONS[function]
        else:
            apply_function = lambda x, y: y

        if refresh_rate:
            plot_refresh_rate = self.update_refresh_rate*(refresh_rate//self.update_refresh_rate)
        else:
            plot_refresh_rate = self.update_refresh_rate*(self.refresh_rate//self.update_refresh_rate)
        
        for channel in channels:
            self.plots[source].append({
                'pos': position,
                'channels': channel,
                'apply_function': apply_function,
                'nth': nth,
                'since_refresh': 1e40,
                'refresh_rate': plot_refresh_rate,
                't_span': t_span,
            })


    def add_image(self, source: str, channel: Union[str, int], function: Optional[Union[str, callable]] = None, refresh_rate: int = 100, colormap: str = 'CET-L17') -> None:
        """Add an image plot to the visualization for the specified source and channel.

        Args:
            source (str): The name of the source to add the image plot to.
            channel (str/int): The name of the channel to add the image plot to.
            function (function/str, optional): A function or string to apply to the image data before plotting. Defaults to None.
            refresh_rate (int, optional): The number of milliseconds between updates of the plot. Defaults to 100.
            colormap (str, optional): The colormap to use for the plot. Defaults to 'CET-L17'.


        This method adds an image plot to the visualization for the specified `source` and `channel`.
        The `function` argument can be used to apply a custom function to the image data before plotting.
        If `function` is not specified or is not a callable function or a string, the identity function is used.
        If `function` is a string, it is looked up in the `INBUILT_FUNCTIONS` dictionary.
        The `refresh_rate` argument specifies the number of milliseconds between updates of the plot.
        The `colormap` argument specifies the colormap to use for the plot.

        If `source` is not already in `self.plots`, a new entry is created for it.
        If `channel` is not already in the entry for `source` in `self.plots`, a new plot is created for it.

        This method modifies the `plots` and `color_map` attributes of the `Visualization` object in-place.
        """
        self.add_image_widget = True

        if self.plots is None:
            self.plots = {}
        
        if source not in self.plots.keys():
            self.plots[source] = []

        if isinstance(function, types.FunctionType):
            apply_function = function
        elif function in INBUILT_FUNCTIONS.keys():
            apply_function = INBUILT_FUNCTIONS[function]
        else:
            apply_function = lambda x, y: y
        

        self.plots[source].append({
            'pos': 'image',
            'channels': channel,
            'apply_function': apply_function,
            'nth': 1,
            'since_refresh': 1e40,
            'refresh_rate': refresh_rate,
            'color_map': colormap,
        })

        self.color_map = colormap


    def config_subplot(self, position: Tuple[int, int], xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None, t_span: Optional[float] = None, axis_style: Optional[str] = 'linear', title: Optional[str] = None, rowspan: int = 1, colspan: int = 1) -> None:
        """Configure a subplot at position `position`.

        Args:
            position (tuple): Tuple of two integers, the position of the subplot in the layout.
            xlim (tuple, optional): Tuple of two floats, the limits of the x-axis. If not given, the limits are set to `(0, 1)`.
            ylim (tuple, optional): Tuple of two floats, the limits of the y-axis. Defaults to None.
            t_span (int/float, optional): The length of the time axis. If this option is not specified, it is computed from the `xlim`.
                Defaults to None.
            axis_style (str, optional): The style of the axis. Can be "linear", "semilogx", "semilogy" or "loglog". Defaults to "linear".
            title (str, optional): The title of the subplot. Defaults to None.
            rowspan (int, optional): The number of rows the subplot spans. Defaults to 1.
            colspan (int, optional): The number of columns the subplot spans. Defaults to 1.


        This method configures a subplot at position `position` with the specified options.
        The `xlim`, `ylim`, `t_span`, `axis_style`, `title`, `rowspan` and `colspan` options are stored in the `subplot_options`
        dictionary of the `Visualization` object.
        """
        self.subplot_options[position] = {}

        if xlim is not None:
            self.subplot_options[position]['xlim'] = xlim
        if ylim is not None:
            self.subplot_options[position]['ylim'] = ylim
        if t_span is not None:
            self.subplot_options[position]['t_span'] = t_span
        if axis_style is not None:
            self.subplot_options[position]['axis_style'] = axis_style
        if title is not None:
            self.subplot_options[position]['title'] = title
        if rowspan is not None:
            self.subplot_options[position]['rowspan'] = rowspan
        if colspan is not None:
            self.subplot_options[position]['colspan'] = colspan

        if not check_subplot_options_validity(self.subplot_options, self.plots):
            raise ValueError("Invalid subplot options. Check the `rowspan` and `colspan` values.")


    def check(self):
        self.positions = list(set([plot['pos'] for plot in [plot for plots in self.plots.values() for plot in plots]]))[::-1]
        self.positions = [_ for _ in self.positions if _ != 'image']

        # Make sure that all subplots have options defined.
        for pos in self.positions:
            if pos not in self.subplot_options.keys():
                self.subplot_options[pos] = {}

        self._check_t_span_and_xlim()
        self._check_added_lines()
        self._check_channels()

    
    def run(self, core):
        self.core = core
        # self.core.is_running_global = False

        self.check()

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


    def _check_channels(self):
        """Convert between channel names and channel indices.
        
        If the `pos` is 'image', check that the `channel` is a string or an intiger. If it is an intiger, convert it to a string.
        If the `pos` is not 'image', check that the `channel` is a string or an intiger. If it is a string, convert it to an intiger.
        """
        for source, plot_channels in self.plots.items():
            acq_index = self.core.acquisition_names.index(source)
            for i, plot_channel in enumerate(plot_channels):
                if plot_channel['pos'] == 'image':
                    if type(plot_channel['channels']) == str:
                        pass
                    elif type(plot_channel['channels']) == int:
                        pass
                    else:
                        raise ValueError("The `channel` must be a string (`channel_name`) or intiger (`channel_index`).")
                else:
                    if type(plot_channel['channels']) == str:
                        channel = plot_channel['channels']
                        self.plots[source][i]['channels'] = self.core.acquisitions[acq_index].channel_names.index(channel)
                    elif type(plot_channel['channels']) == int:
                        pass
                    elif type(plot_channel['channels']) == tuple:
                        channel1, channel2 = plot_channel['channels']
                        channel1 = self.core.acquisitions[acq_index].channel_names.index(channel1) if type(channel1)==str else channel1
                        channel2 = self.core.acquisitions[acq_index].channel_names.index(channel2) if type(channel2)==str else channel2
                        self.plots[source][i]['channels'] = (channel1, channel2)
                    else:
                        raise ValueError("The `channel` must be a string (`channel_name`), intiger (`channel_index`) or tuple of two strings or intigers.")


    def _check_added_lines(self):
        if self.plots is None:
            raise ValueError("No plots were added to the visualization. Use the `add_lines` method to add plots.")

        n_lines = sum([len(plot_channels) for plot_channels in self.plots.values()])
        
        if hasattr(self, "core"):
            # Determine the nth value for each line.
            for source, plot_channels in self.plots.items():
                acq_index = self.core.acquisition_names.index(source)
                sample_rate = self.core.acquisitions[acq_index].sample_rate
                for i, plot_channel in enumerate(plot_channels):
                    if plot_channel['nth'] == 'auto':
                        t_span = plot_channel['t_span']
                        self.plots[source][i]['nth'] = compute_nth(self.max_points_to_refresh, t_span, n_lines, sample_rate)


    def _check_t_span_and_xlim(self):
        """Check and set the `t_span` and `xlim` options for all plots in `self.plots`.

        If `t_span` is not defined for a plot, it is copied from the corresponding `subplot_options`.
        If `t_span` is not defined in `subplot_options`, it is computed from the `xlim` option.
        If `xlim` is not defined in `subplot_options`, it is set to `(0, 1)` and `t_span` is computed from it.

        This method modifies the `t_span` and `xlim` options in-place for all plots in `self.plots`.
        """
        for source, plot_channels in self.plots.items():
            for i, plot_channel in enumerate(plot_channels):
                if 't_span' in plot_channel.keys(): # image plots don't have t_span
                    if plot_channel['t_span'] is None: # if t_span is None, compute it from xlim or overwrite it with t_span from subplot_options
                        if 't_span' in self.subplot_options[plot_channel['pos']]:
                            plot_channel['t_span'] = self.subplot_options[plot_channel['pos']]['t_span']
                        elif 't_span' not in self.subplot_options[plot_channel['pos']] and 'xlim' in self.subplot_options[plot_channel['pos']]:
                            plot_channel['t_span'] = self.subplot_options[plot_channel['pos']]['xlim'][1] - self.subplot_options[plot_channel['pos']]['xlim'][0]
                        else:
                            plot_channel['t_span'] = 1
        
        # check if xlim is defined for all subplots, if not, compute it from t_span
        for pos, options in self.subplot_options.items():
            if 'xlim' not in options.keys():
                # get max t_span from self.plots for this position
                t_spans = [plot_channel['t_span'] for source, plot_channels in self.plots.items() for plot_channel in plot_channels if plot_channel['pos'] == pos and plot_channel['t_span'] is not None]
                if t_spans:
                    t_span_max = max(t_spans)
                else:
                    t_span_max = 1
                self.subplot_options[pos]['xlim'] = (0, t_span_max)


    def create_ring_buffers(self):
        """Create and initialize the ring buffers for all plots in `self.plots`.

        For each source in `self.plots`, this method creates a `RingBuffer2D` object with the appropriate number of rows
        and channels, based on the `t_span` and `sample_rate` options in `self.plots` and the corresponding acquisition.
        If the acquisition has video channels, this method also initializes a list of random images for each video channel.
        If a source does not have any channels, a `RingBuffer2D` object with one row and one channel is created.

        This method modifies the `ring_buffers` and `new_images` attributes of the `Visualization` object in-place.
        """
        self.ring_buffers = {}
        for source in self.plots.keys():
            acq = self.core.acquisitions[self.core.acquisition_names.index(source)]
            if acq.channel_names:
                n_channels = len(acq.channel_names)
                # rows = int(max([self.subplot_options[pos]['t_span'] * acq.sample_rate for pos in self.positions]))
                rows = int(max([_['t_span'] * acq.sample_rate for _ in self.plots[source] if _['pos'] != 'image']))
                self.ring_buffers[source] = RingBuffer2D(rows, n_channels)
            
            if acq.channel_names_video:
                # self.new_images = [np.random.rand(10, 10)] * len(acq.channel_names_video)
                self.new_images = [(ch, np.random.rand(10, 10)) for ch in acq.channel_names_video]

            if source not in self.ring_buffers.keys():
                self.ring_buffers[source] = RingBuffer2D(1, 1)


class MainWindow(QMainWindow):
    def __init__(self, vis, core, app):
        super().__init__()
        
        self.vis = vis
        self.core = core
        self.app = app

        script_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(script_directory, "../logo.png")
        app_icon = QIcon(icon_path)
        self.setWindowIcon(app_icon)

        self.triggered = False
        self.measurement_stopped = False
        self.freeze_plot = False

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
        # Compute the update refresh rate
        n_lines = sum([len(plot_channels) for plot_channels in self.vis.plots.values()])
        minimum_refresh_rate = int(min(list(set([plot['refresh_rate'] for plot in [plot for plots in self.vis.plots.values() for plot in plots]]))))
        
        # Compute the max number of plots per refresh (if sequential plot updates are enabled)
        if self.vis.sequential_plot_updates:
            # Max number of plots per refresh is computed
            computed_update_refresh_rate = max(10, min(500, int(minimum_refresh_rate/(n_lines+1))))
            self.vis.max_plots_per_refresh = int(np.ceil((n_lines * computed_update_refresh_rate) / minimum_refresh_rate))
            self.vis.update_refresh_rate = computed_update_refresh_rate
        else:
            self.vis.max_plots_per_refresh = 1e40
            self.vis.update_refresh_rate = minimum_refresh_rate


        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.time_start = time.time()
        grid_layout = pg.GraphicsLayoutWidget()

        self.subplots = {}
        self.legends = {}

        if self.vis.add_line_widget:
            self.layout_widget.addWidget(grid_layout, stretch=1)

        color_dict = {}
        ##################################################################
        
        # Create subplots
        for pos in self.vis.positions:
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

                self.subplots.update({pos: grid_layout.addPlot(*pos, rowspan=rowspan, colspan=colspan, title=title)})

                if pos in self.vis.subplot_options.keys():
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
                
        # Create lines for each plot channel
        images = 0
        for source, plot_channels in self.vis.plots.items():
            channel_names = self.core.acquisitions[self.core.acquisition_names.index(source)].channel_names
            color_dict.update({ch: ind+len(color_dict) for ind, ch in enumerate(channel_names)})

            for i, plot_channel in enumerate(plot_channels):
                pos = plot_channel['pos']
                ch = plot_channel['channels']

                if pos == 'image':
                    images += 1
                    if "boxstate" in plot_channel.keys():
                        # remove the key
                        plot_channel.pop("boxstate")

                    if plot_channel['color_map'] == 'CET-L17':
                        cm = pg.colormap.get(plot_channel['color_map'])
                    else:
                        cm = pg.colormap.getFromMatplotlib(plot_channel['color_map'])

                    if cm.color[0, 0] == 1:
                        cm.reverse()

                    image_view = HoverImageView()
                    image_view.setColorMap(cm)

                    if images == 1:
                        self.image_grid_layout = QGridLayout()
                        self.layout_widget.addLayout(self.image_grid_layout, stretch=1)
                    
                    col, row = divmod(images-1, 2)
                    self.image_grid_layout.addWidget(image_view, row, col)
                    image_view.ui.histogram.hide()
                    image_view.ui.roiBtn.hide()
                    image_view.ui.menuBtn.hide()
                    self.vis.plots[source][i]['image_view'] = image_view
                else:
                    if isinstance(ch, tuple):
                        x, y = ch
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[y]], width=2), name=f"{channel_names[x]} vs. {channel_names[y]}")
                        self.vis.plots[source][i]['line'] = line

                    elif isinstance(ch, int):
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=color_dict[channel_names[ch]], width=2), name=f"{channel_names[ch]}")
                        self.vis.plots[source][i]['line'] = line

                    # Add legend to the subplot
                    if pos not in self.legends.keys() and pos != 'image':
                        legend = self.subplots[pos].addLegend()
                        for item in self.subplots[pos].items:
                            if isinstance(item, pg.PlotDataItem):
                                legend.addItem(item, item.opts['name'])
                        self.legends[pos] = legend

        self.plots = self.vis.plots
        

    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.vis.update_refresh_rate)


    def update_ring_buffers(self):
        for source, buffer in self.vis.ring_buffers.items():
            acq = self.core.acquisitions[self.core.acquisition_names.index(source)]
            if acq.channel_names_video:
                plot_channel = self.plots[source][-1]
                since_refresh = plot_channel['since_refresh']
                refresh_rate = plot_channel['refresh_rate']
                if (refresh_rate <= since_refresh + self.vis.update_refresh_rate):
                    _, new_data = acq.get_data(N_points=1, data_to_return="video")
                    # self.new_images = [_[-1].T for _ in new_data]
                    self.new_images = dict([(ch, _[-1].T) for ch, _ in zip(acq.channel_names_video, new_data)])

            if acq.channel_names:
                new_data = acq.get_data_PLOT()
                buffer.extend(new_data)


    def update_plots(self, force_refresh=False):
        # Stop the measurement if the acquisitions are done and if the measurement has not been stopped.
        if not self.core.is_running_global and not self.measurement_stopped:
            self.stop_measurement()

        # If the emasurement is started, start the timer and update the progress bar.
        if self.core.triggered_globally and not self.triggered:
            self.on_measurement_start()

            if self.core.measurement_duration is not None: 
                self.progress_bar.setMaximum( 1000 ) 
            else:
                pass

        # If the measurement is running, update the progress bar and the label.
        if self.triggered and self.core.is_running_global:
            if self.core.measurement_duration is not None:
                
                progress_value = float(self.core.acquisitions[0].Trigger.N_acquired_samples_since_trigger)/float(self.core.acquisitions[0].Trigger.N_samples_to_acquire)*1000
                self.progress_bar.setValue( int(progress_value) )
                string = f"Duration: {self.core.measurement_duration:.1f} s"
            else:
                string = "Duration: Until stopped"
            self.label.setText(string) 

        # Update the ring buffers.
        self.update_ring_buffers()

        if not self.freeze_plot:
            updated_plots = 0
            for source, plot_channels in self.plots.items():
                self.vis.acquisition = self.core.acquisitions[self.core.acquisition_names.index(source)]

                # for line, pos, apply_function, *channels in plot_channels:
                for i, plot_channel in enumerate(plot_channels):
                    refresh_rate = plot_channel['refresh_rate']
                    since_refresh = plot_channel['since_refresh']

                    if (refresh_rate <= since_refresh + self.vis.update_refresh_rate or force_refresh) and updated_plots < self.vis.max_plots_per_refresh:
                        # If time to refresh, refresh the plot and set since_refresh to 0.
                        plot_channel['since_refresh'] = 0
                        
                        if plot_channel['pos'] == 'image':
                            if hasattr(self, 'new_images'):
                                new_data = self.new_images[plot_channel['channels']]
                                #print(new_data.shape)
                                self.update_image(new_data, plot_channel)
                        else:
                            new_data = self.vis.ring_buffers[source].get_data()
                            self.update_line(new_data, plot_channel)

                        updated_plots += 1
                    else:
                        # If not time to refresh, increase since_refresh by update_refresh_rate.
                        plot_channel['since_refresh'] += self.vis.update_refresh_rate
    
    
    def update_image(self, new_data, plot_channel):
        _view = plot_channel['image_view'].getView()
        if 'boxstate' in plot_channel.keys():
            _state = _view.getState()

        plot_channel['image_view'].setImage(new_data)

        if 'boxstate' in plot_channel.keys():
            _view.setState(_state)
        
        plot_channel['boxstate'] = True


    def update_line(self, new_data, plot_channel):
        # only plot data that are within xlim (applies only for normal plot, not ch vs. ch)
        t_span_samples = int(plot_channel['t_span'] * self.vis.acquisition.sample_rate)
        
        nth = plot_channel['nth']

        xlim = self.vis.subplot_options[plot_channel['pos']]['xlim']

        if isinstance(plot_channel['channels'], int):
            # plot a single channel
            ch = plot_channel['channels']
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

        elif isinstance(plot_channel['channels'], tuple): 
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
        #self.core.triggered_globally = True # dummy start measurement
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

        for pos, legend in self.legends.items():
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


        