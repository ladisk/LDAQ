Getting started
============

Acquisition
-----------
First, define the acquisition and generation (optional) objects. Here, the example is shown for National Instruments acquisition
and generation.

Acquisition object is defined:
::

    acq = LadiskDAQ.NIAcquisition(input_task_name)

To start the acquisition, a trigger must be set:
::

    acq.set_trigger(level=100, channel=0, duration=11, presamples=10)

The acquisition can now be started on its own by running:
::

    acq.run_acquisition()

Generation
----------
Optionally, the ``generation`` object can also be created:
::

    gen = LadiskDAQ.NIGenerator(output_task_name, signal)

where the ``signal`` is a ``numpy`` array. Each row of the array represents one output channel.

Similar to acquisition, the generatino can be strated on its own by running:
::

    gen.run_generation()

Visualization, synchronization and triggering
---------------------------------------------
To visualize the acquired signals during the measurement, the ``acq`` object is passed to ``LDAQ`` class:
::

    ldaq = LadiskDAQ.LDAQ(acq)

To costumize the visualization:
::

    ldaq.configure(plot_layout='default', max_time=5.0, nth_point='auto', autoclose=True, refresh_interval=0.01)

see the `docstring <https://github.com/ladisk/LadiskDAQ/blob/84d574cfa8c5ccaab991a13fda5de56bc9509b0e/LadiskDAQ/core.py#L35>`_ for further details.
For ``plot_layout`` details, see :ref:`below<Plot Layout section>`.

To start the acquisition and visualization, call:
::

    ldaq.run()

The ``.run()`` method can be called sequentially, without having to run the ``acq`` or ``ldaq`` setup.

Additionally, the ``gen`` object can also be included to output the signal during the acquisition. To include the generation:
::

    ldaq = LadiskDAQ.LDAQ(acq, gen)

Again, the ``.run()`` mehtod can be called sequentially (for example in a for loop).

During the acquisition, some user interaction is supported. See below for the :ref:`supported hotkeys<Hotkeys section>`.


Saving the measurement
----------------------
After the acquisition is done, the measurement can be saved by calling:
::

    ldaq.save_measurement(name, root='', save_channels='All', timestamp=True, comment='')

or

::

    acq.save(name, root='', save_channels='All', timestamp=True, comment='')

Both methods are equal, the second one must be used in the acquisition was started on its own (without creating the ``ldaq`` object).

.. _Plot Layout section:

Plot layout
-----------
Normal time plots
~~~~~~~~~~~~~~~~~

With plot layout, the user can define on which subplot the channels will be plotted. An example of plot layout is:

::
    
    plot_layout = {
        (0, 0): [0, 1],
        (0, 1): [2, 3]
        }

On the first subplot (0, 0), channels 0 and 1 will be plotted.
On the second subplot (0, 1), channels 2 and 3 will be plotted.

Channel vs. channel
~~~~~~~~~~~~~~~~~~~
If, for example, we wish to plot channel 1 as a function of channel 0, input
channel indices in a tuple; first the channel to be plotted on x-axis, and second the channel to be plotted on y-axis:
::

    plot_layout = {
            (0, 0): [0, 1],
            (0, 1): [2, 3],
            (1, 0): [(0, 1)]
        }

Fourier transform
~~~~~~~~~~~~~~~~~
The DFT of the signal can be computed on the fly. To define the subplots and channels where the FFT is computed, 
add "fft" as an element into channel list. Additionaly 'logy' and 'logx' scalings can be set:
::

    plot_layout = {
            (0, 0): [0, 1],               # Time series
            (0, 1): [2, 3],               # Time series
            (1, 0): [(0, 1)],             # ch1 = f( ch0 )
            (1, 1): [2, 3, "fft", "logy"] # FFT(2) & FFT(3), log y scale
        }

Custom function plot
~~~~~~~~~~~~~~~~~~~~
Lastly, the signals can be modified for visualization by specifying a custom function, that is passed to the channel list.
Example below computes the square of the signal coming from channels 1 and 2. 
::

    plot_layout = {
            (0, 0): [0, 1],               # Time series
            (0, 1): [2, 3],               # Time series
            (1, 0): [(0, 1)],             # ch1 = f( ch0 )
            (1, 1): [2, 3, fun]           # fun(2) & fun(3)
        }

Function definition example:
::

    def fun(self, channel_data):
        '''
        :param self:         instance of the acquisition object (has to be there so the function is called properly)
        :param channel_data: channel data
        '''
        return channel_data**2


Custom function plot - channel vs. channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    plot_layout = {
            (0, 0): [(0, 1), fun]         # 2Darray = fun( np.array([ch0, ch1]).T )
        }

Function definition examples:
::

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


.. _Hotkeys section:

Supported hotkeys
-----------------

The supported hotkeys. Press these keys for the desired action while the plot is active.

+--------+--------------------------------------------------+
| HotKey | Action                                           |
+========+==================================================+
| q      | Stop the measurement                             |
+--------+--------------------------------------------------+
| s      | Start the measurement manually (without trigger) |
+--------+--------------------------------------------------+
| f      | Freeze the plot during the measurement           |
+--------+--------------------------------------------------+
| Space  | Resume the plot after freeze                     |
+--------+--------------------------------------------------+