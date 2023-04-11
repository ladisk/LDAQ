import numpy as np
import types

# def auto_nth_point(plot_layout, max_time, sample_rate, max_points_to_refresh=1e5):
#     """
#     Automatically determine the skip interval for drawing points.
#     """
#     lines = 0
#     for key, val in plot_layout.items():
#         for k, v in val.items():
#             for i in v:
#                 if type(i) in [tuple, int]:
#                     lines += 1
    
#     points_to_refresh = max_time*sample_rate*lines
    
#     if max_points_to_refresh < points_to_refresh:
#         nth = int(np.ceil(points_to_refresh/max_points_to_refresh))
#     else:
#         nth = 1
#     return nth

def auto_nth_point(layout, subplot_options, core, max_points_to_refresh):
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

    Example of the `layout` dictionary:
    layout = {
        'NI_task': {
            (0, 0): [0, 1],
            (1, 0): [(0, 1)],
        },
        'Arduino_1': {
            (0, 1): [0, 1, _fun_fft],
            (1, 1): [0, 1],
        }
    }
    (where _fun_fft is a function that is applied to the data before plotting. 
    It should not be taken into account when calculating `nth`).

    Example of the `subplot_options` dictionary:
    subplot_options = {
        (0, 0): {
            'xlim': (0, 2),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        },
        (0, 1): {
            'xlim': (0, 25),
            'ylim': (1e-5, 10),
            'axis_style': 'semilogy'
        },
        (1, 0): {
            'xlim': (-5, 5),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        },
        (1, 1): {
            'xlim': (0, 2),
            'axis_style': 'linear'
        }
    }
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
                    
                    if 'xlim' in subplot_options[subplot]:
                        if subplot in subplot_options:
                            xlim = subplot_options[subplot]['xlim']
                        else:
                            for subplot2, channels2 in acq_layout.items():
                                if channel in channels2:
                                    xlim = subplot_options[subplot2]['xlim']
                                    break
                    else:
                        xlim = (0, 1)

                    acq_index = core.acquisition_names.index(acq_name)
                    sample_rate = core.acquisitions[acq_index].sample_rate
                    nth[acq_name][subplot][channel] = int(np.ceil(sample_rate * (xlim[1] - xlim[0]) / (max_points_to_refresh/get_nr_of_lines(layout))))
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

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return np.array([freq, np.abs(amp)]).T

