import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import sys
import random
import time
import types

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


INBUILT_FUNCTIONS = {'fft': _fun_fft, 'frf_amp': _fun_frf_amplitude, 'frf_phase': _fun_frf_phase}


class Visualization:
    def __init__(self, layout=None, subplot_options=None):
        self.layout = layout
        self.subplot_options = subplot_options
        

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
        self.app = app
        self.core = core

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
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()))
                        plot_channels.append((line, apply_function, x, y))
                    elif isinstance(ch, int):
                        line = self.subplots[pos].plot(pen=pg.mkPen(color=random_color()))
                        plot_channels.append((line, apply_function, ch))
                    elif isinstance(ch, types.FunctionType):
                        pass

            self.plots[source] = plot_channels


    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)


    def update_plots(self):
        if not self.core.is_running_global:
            self.close_app()

        new_data = self.core.get_measurement_dict(2)
        for source, plot_channels in self.plots.items():
            self.vis.acquisition = self.core.acquisitions[self.core.acquisition_names.index(source)]

            for line, apply_function, *channels in plot_channels:
                if len(channels) == 1: # plot a single channel
                    ch = channels[0]
                    fun_return = apply_function(self.vis, new_data[source]["data"][:, ch])
                    if len(fun_return.shape) == 1: # if function returns only 1D array
                        y = fun_return
                        x = new_data[source]["time"]
                    else:  # function returns 2D array (e.g. fft returns freq and amplitude)
                        x, y = fun_return.T # expects 2D array to be returned

                    line.setData(x, y)

                else: # channel vs. channel
                    channel_x, channel_y = channels
                    fun_return = apply_function(self.vis, new_data[source]['data'][:, [channel_x, channel_y]])
                    x, y = fun_return.T
                    # x_ch, y_ch = channels
                    # x = new_data[source]['data'][:, x_ch]
                    # y = new_data[source]['data'][:, y_ch]

                    line.setData(x, y)



    def close_app(self):
        self.timer.stop()
        self.app.quit()
        self.close()

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))



