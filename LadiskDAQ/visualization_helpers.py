import numpy as np
from scipy.signal import coherence
import types


def auto_nth_point(layout, subplot_options, core, max_points_to_refresh, known_nth='auto'):
    """For each channel in `layout` find the `nth` variable that says that every nth point
    is plotted.
    
    `nth` is determined for each line, based on the `xlim` parameter in the `subplot_options` if `xlim`
    is a key in the `subplot_options` dictionary. If `xlim` is not a key in the `subplot_options` dictionary
    take the `xlim` from the `subplot_options` of the subplot where the line is plotted.
    `core` is the object of the `LDAQ.Core` class. It is used to get the `acquisition` object
    where the `sample_rate` is taken from.

    :param layout: layout of the QT application. It specifies which channels are plotted on each subplot.
    :param subplot_options: Options for each subplot (xlim, ylim, axis_style, etc.)
    :param core: instance of the `LDAQ.Core` class.
    :param max_points_to_refresh: maximum number of points to refresh in one refresh cycle.
    """
    nth = {}
    for acq_name, acq_layout in layout.items():
        nth[acq_name] = {}
        for subplot, channels in acq_layout.items():
            nth[acq_name][subplot] = {}
            for channel in channels:
                if not isinstance(channel, (types.FunctionType, str)):
                    if isinstance(channel, tuple):
                        channel = channel[0]

                    if 'nth' in subplot_options[subplot].keys():
                        nth[acq_name][subplot][channel] = subplot_options[subplot]['nth']

                    else:
                        if known_nth == 'auto':
                        
                            if 't_span' in subplot_options[subplot]:
                                if subplot in subplot_options:
                                    t_span = subplot_options[subplot]['t_span']
                                else:
                                    for subplot2, channels2 in acq_layout.items():
                                        if channel in channels2:
                                            t_span = subplot_options[subplot2]['t_span']
                                            break
                            else:
                                t_span = 1

                            acq_index = core.acquisition_names.index(acq_name)
                            sample_rate = core.acquisitions[acq_index].sample_rate
                        
                            points_per_line = max_points_to_refresh/get_nr_of_lines(layout)
                            nth[acq_name][subplot][channel] = int(np.ceil(sample_rate * (t_span) / points_per_line))

                        elif isinstance(known_nth, int):
                            nth[acq_name][subplot][channel] = known_nth
                        else:
                            raise ValueError('`known_nth` should be either "auto" or an integer.')
    return nth

def get_nr_of_lines(layout):
    """get the number of lines in all plots, based on the layout"""
    nr_of_lines = 0
    for acq_name, acq_layout in layout.items():
        for subplot, channels in acq_layout.items():
            for channel in channels:
                if not isinstance(channel, (types.FunctionType, str)):
                    nr_of_lines += 1
    return nr_of_lines


def check_subplot_options_validity(subplot_options, layout):
    """
    Check if the plot layout is valid with the given rowspan and colspan.

    :param subplot_options: Options for each subplot (xlim, ylim, axis_style, etc.)
    :param layout: layout of the QT application. It specifies which channels are plotted on each subplot.
    :return: True if the layout is valid, False otherwise.
    """
    max_row = max([pos[0] for acq_name, acq_layout in layout.items() for pos, channels in acq_layout.items()])
    max_col = max([pos[1] for acq_name, acq_layout in layout.items() for pos, channels in acq_layout.items()])

    # Create a matrix to keep track of the occupied cells in the subplot grid
    occupied_cells = [[False] * (max_col + 1) for _ in range(max_row + 1)]

    # Loop through each subplot and check if it is valid
    for pos, options in subplot_options.items():
        row, col = pos
        rowspan = options.get('rowspan', 1)
        colspan = options.get('colspan', 1)

        # Check if the subplot is out of bounds
        if row + rowspan - 1 > max_row or col + colspan - 1 > max_col:
            return False

        # Check if the subplot overlaps with another subplot
        for i in range(row, row + rowspan):
            for j in range(col, col + colspan):
                if occupied_cells[i][j]:
                    return False
                occupied_cells[i][j] = True

    # If all subplots are valid, return True
    return True

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return np.array([freq, np.abs(amp)]).T

class _FRF_calculation():
    """
    This class can be used for averaging the FRF over multiple calls of the same function.
    """
    def __init__(self) -> None:
        self.first_call = True
        self.N_samples = 0
        self.freq = None
        self.H1 = None
        self.n_avg = 10
        self.n_calls = 1
        
        self.last_fun_call = None
    
    def _calc_frf(self, self_vis, channel_data):   
        if self_vis.acquisition.is_ready == False: # this means that was run for the first time
            self.first_call = True
        
        # estimate FRF:
        x, y = channel_data.T
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        Sxy = np.conj(X) * Y
        Sxx = np.conj(X) * X
        H1 = Sxy / Sxx
        
        if self.first_call: # create the frequency vector and H1
            self.N_samples = len(channel_data)
            self.freq = np.fft.rfftfreq(self.N_samples, d=1/self_vis.acquisition.sample_rate)
            self.first_call = False
            self.H1 = H1
        else: # only update H1
            n = self.n_calls if self.n_calls < self.n_avg else self.n_avg
            
            self.H1 = self.H1 * (n-1)/n + H1 * 1/n
            
        self.n_calls += 1
    
    def get_frf_abs(self, self_vis, channel_data):
        self._calc_frf(self_vis, channel_data) # always update the FRF
        self.last_fun_call = 'abs'
        return np.array([self.freq, np.abs(self.H1)]).T
    
    def get_frf_phase(self, self_vis, channel_data):
        if self.last_fun_call == 'phase' or self.last_fun_call is None: # only update if the last call was phase
            self._calc_frf(self_vis, channel_data)
        self.last_fun_call = 'phase'
        return np.array([self.freq, np.angle(self.H1)*180/np.pi]).T
    
def _fun_frf_amp(self_vis, channel_data):   
    """Default function for calculating the FRF amplitude.

    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.

    Returns:
        2D numpy array: np.array([freq, np.abs(H1)]).T
    """
    # estimate FRF:
    x, y = channel_data.T
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    Sxy = np.conj(X) * Y
    Sxx = np.conj(X) * X
    H1 = Sxy / Sxx
    
    freq = np.fft.rfftfreq(len(channel_data), d=1/self_vis.acquisition.sample_rate)
    return np.array([freq, np.abs(H1)]).T
    
def _fun_frf_phase(self_vis, channel_data):   
    """Default function for calculating the FRF phase.
    
    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.
        
    Returns:
        2D numpy array: np.array([freq, np.angle(H1)*180/np.pi]).T
    """
    # estimate FRF:
    x, y = channel_data.T
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    Sxy = np.conj(X) * Y
    Sxx = np.conj(X) * X
    H1 = Sxy / Sxx
    
    freq = np.fft.rfftfreq(len(channel_data), d=1/self_vis.acquisition.sample_rate)
    return np.array([freq, np.angle(H1)*180/np.pi]).T

def _fun_coh(self_vis, channel_data):   
    """Default function for calculating the coherence.
    
    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.
        
    Returns:
        2D numpy array: np.array([freq, coherence]).T
    """
    # estimate FRF:
    x, y = channel_data.T
    fs   = self_vis.acquisition.sample_rate
    # X = np.fft.rfft(x)
    # Y = np.fft.rfft(y)
    # Sxy = np.conj(X) * Y
    # Sxx = np.conj(X) * X
    # Syy = np.conj(Y) * Y
    
    # coh = np.abs( ( np.abs(Sxy)**2 / (Sxx * Syy) ) )
    # coh[ np.isnan(coh) ] = 0
    
    # freq = np.fft.rfftfreq(len(channel_data), d=1/fs)
    
    freq, coh = coherence(x, y, fs, nperseg=fs*2)
    
    return np.array([freq, coh]).T
        
        
        
        
        
    

