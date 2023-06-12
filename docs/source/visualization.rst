Live visualization
==================

Live visualization of the measurments is possible by adding the :class:`LadiskDAQ.Visualization` object to the
:class:`LadiskDAQ.Core` class:

.. code-block:: python

    ldaq = LadiskDAQ.Core(acq, visualization=vis)


- ``acq`` is defined as presented in the `first example <simple_start.html>`_. 
- ``vis`` is an instance of :class:`LadiskDAQ.Visualization` and is initiated as follows:

.. code-block:: python

    vis = LadiskDAQ.Visualization(refresh_rate=100, max_points_to_refresh=1e4, sequential_plot_updates=True)

- ``refresh_rate``: defines the refresh rate of the live plot in milliseconds. ``refresh_rate`` can also be defined for each 
  line/image separetely (see :ref:`add_lines <add_lines>` for more details).
- ``max_points_to_refresh``: defines the maximum number of points to refresh in the plot. Adjust this number to optimize performance.
  This number is used to compute the ``nth`` value automatically.
- ``sequential_plot_updates``: if ``True``, the lines are updated sequentially in each iteration of the main loop. 
  Potentially, the plot can show a phase shift between the lines. This is because when the first line is updated, 
  the data for the second line is already acquired. To avoid this, set ``sequential_plot_updates`` to ``False``.

.. _add_lines:
Adding lines to the plot
------------------------

To show plot the data, the lines have to be added to the plot. The number of subplots is also determined in this step.

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=0, function=None, nth=10, refresh_rate=1000)

The arguments are:

- ``position``: the position of the subplot in the plot. The position is defined as a tuple of two integers, where the first integer is the row number and the second integer is the column number of the subplot.
- ``source``: the name of the acquisition source. The name is defined in the acquisition class.
- ``channels``: the indices of the channels to be plotted. See :ref:`channels <channels_argument>` for more details.
- ``function``: the function to be applied to the data. This can be a built-in function or a custom function. See :ref:`the function argument <function_argument>` for more details.
- ``nth``: the number of data points to be plotted. If ``nth`` is set to 10, every 10th data point will be plotted. This is useful when the data is acquired at a high sample rate and the plot is updated at a low refresh rate.
  By default, ``nth`` is set to ``'auto'``. In this case, the number of data points to be plotted is determined automatically.
- ``refresh_rate``: the refresh rate of the subplot. If ``None``, the refresh rate is set to the refresh rate set in the ``Visualization`` object.
- ``t_span``: the time span of the data to be plotted. If ``None``, the time span is computed based on the ``xlim``. The ``t_span`` defines the length of the data passed to a function.
  This is the same argument that can be given in the ``config_subplot`` method, but if it is defined here, this is the one that is taken into account.


.. _channels_argument:
Channels
~~~~~~~~

If the ``channels`` argument is an integer, the data from the channel with the specified index will be plotted.

If the ``channels`` argument is a list of integers, the data from the channels with the specified indices will be plotted:

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=[0, 1])

To plot channel vs. channel the ``channels`` argument is a tuple of two integers:

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=(0, 1))

The first integer is the index of the x-axis and the second integer is the index of the y-axis.

Multiple channel vs. channel plots can be added to the same subplot:

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=[(0, 1), (2, 3)])

.. _function_argument:
The ``function`` argument
~~~~~~~~~~~~~~~~~~~~~~~~

The data can be processed on-the-fly by a specified function.


The ``function`` can be specified by the user. To use the built-in functions, a string is passed to the ``function`` argument. 
An example of a built-in function is "fft" which computes the `Fast Fourier Transform <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html>`_ 
of the data with indices 0 and 1:

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=[0, 1], function='fft')

To build a custom function, the function must be defined as follows:

.. code-block:: python

    def function(self, channel_data):
        '''
        :param self: instance of the acquisition object (has to be there so the function is called properly)
        :param channel_data: channel data
        '''
        return channel_data**2

The ``self`` argument in the custom function referes to the instance of the acquisition object. 
This connection can be used to access the properties of the acquisition object, e.g. sample rate.
The ``channel_data`` argument is a list of numpy arrays, where each array corresponds to the data from one channel. 
The data is acquired in the order specified in the ``channels`` argument.

For the example above, the custom function is called for each channel separetely, the ``channel_data`` is a one-dimensional numpy array. 
To add mutiple channels to the ``channel_data`` argument, the ``channels`` argument is modified as follows:

.. code-block:: python

    vis.add_lines(position=(0, 0), source='DataSource', channels=[(0, 1)], function=function)

The ``function`` is now passed the ``channel_data`` with shape ``(N, 2)`` where ``N`` is the number of samples.
The function can also return a 2D numpy array with shape ``(N, 2)`` where the first column is the x-axis and the second column is the y-axis.
An example of such a function is:

.. code-block:: python

    def function(self, channel_data):
        '''
        :param self: instance of the acquisition object (has to be there so the function is called properly)
        :param channel_data: 2D channel data array of size (N, 2)

        :return: 2D array np.array([x, y]).T that will be plotted on the subplot.
        '''
        ch0, ch1 = channel_data.T

        x =  np.arange(len(ch1)) / self.acquisition.sample_rate # time array
        y = ch1**2 + ch0 - 10

        return np.array([x, y]).T


.. _config_subplots:
Configure the subplots
----------------------

To configure the subplots, the ``config_subplot`` method is used:

.. code-block:: python

    vis.config_subplot(position=(2, 2), xlim=None, ylim=None, t_span=None, axis_style='linear', title=None, rowspan=1, colspan=1)

The valid arguments are:

- ``position``: the position of the subplot in the plot. 
- ``xlim``: the x-axis limits of the subplot. If ``None``, the limits are set to ``(0, 1)``.
- ``ylim``: the y-axis limits of the subplot. If ``None``, the limits are set automatically.
- ``t_span``: the time span of the data to be plotted. If ``None``, the time span is computed based on the ``xlim``. The ``t_span`` defines the length of the data passed to a function.
- ``axis_style``: the style of the axis. The valid options are ``'linear'``, ``'semilogy'``, ``'semilogx'`` and ``'loglog'``.
- ``title``: the title of the subplot.
- ``rowspan``: the number of rows the subplot spans.
- ``colspan``: the number of columns the subplot spans.

.. note:: 
    When plotting a simple time signal, the ``t_span`` and ``xlim`` have the same effect. 
    
    However, when plotting channel vs. channel, the ``t_span`` specifies the time range of the data and the ``xlim`` specifies the range of the x-axis (spatial).

    When plotting a function, the ``t_span`` determines the time range of the data that is passed to the function. 
    Last ``t_span`` seconds of data are passed to the function.


.. note::
    The ``xlim`` defines the samples that are plotted on the x-axis, not only a narrowed view of the data. 
    With this, the same data can be viewed with different zoom levels in an effcient way.


