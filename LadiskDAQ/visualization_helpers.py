import numpy as np


def auto_nth_point(plot_layout, max_time, sample_rate, max_points_to_refresh=1e5):
    """
    Automatically determine the skip interval for drawing points.
    """
    lines = 0
    for key, val in plot_layout.items():
        for k, v in val.items():
            for i in v:
                if type(i) in [tuple, int]:
                    lines += 1
    
    points_to_refresh = max_time*sample_rate*lines
    
    if max_points_to_refresh < points_to_refresh:
        nth = int(np.ceil(points_to_refresh/max_points_to_refresh))
    else:
        nth = 1
    return nth

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return np.array([freq, np.abs(amp)]).T

