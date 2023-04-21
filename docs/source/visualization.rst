Live visualization
==================

Live visualization of the measurments is possible by adding the :class:`LadiskDAQ.Visualization` object to the
:class:`LadiskDAQ.Core` class:

.. code-block:: python

    ldaq = LadiskDAQ.Core(acq, visualization=vis)


- ``acq`` is defined as presented in the `first example <simple_start.html>`_. 
- ``vis`` is an instance of :class:`LadiskDAQ.Visualization` and is initiated as follows:

.. code-block:: python

    vis = LadiskDAQ.Visualization(
        layout=layout, 
        subplot_options=subplot_options, 
        nth="auto", 
        refresh_rate=10
    )

- ``layout``: dictionary that defines the layout of the live plot. See the :ref:`layout section <layout>` for more details.
- ``subplot_options``: set the properties of each subplot, defined with the ``layout`` argument. See the :ref:`subplot_options section <subplot_options>` for more details.
- ``nth``: defines the number of samples that are plotted. If ``nth`` is set to "auto", the number of samples is automatically determined based on the number of channels and the sample rate of each acquisition source. The effect of ``nth`` is that every ``nth`` sample is plotted.
- ``refresh_rate``: defines the refresh rate of the live plot in milliseconds.

.. _layout:
The ``layout``
--------------

The layout of the live plot is set by the ``layout`` argument. An example of the ``layout`` argument is:

.. code-block:: python

    layout = {
        'DataSource': {
            (0, 0): [0, 1],
            (1, 0): [2, 3],
        }
    }

This is a layout for a single acquisition source with name "DataSource". When multiple sources are used, the name of the source is used as the key in the ``layout`` dictionary. 
The value at each acquisition source is a dictionary where each key is a tuple of two integers. The first integer is the row number and the second integer is the column number of the subplots.

For the given example, the plot will have two subplots, each in one row.

For each subplot, the data is then specified. If the value is a list of integers, each integer corresponds to the index in the acquired data.
For example, for the subplot defined with:

.. code-block:: python

    (0, 0): [0, 1]

data with indices 0 and 1 will be plotted in the subplot at location (0, 0).

Plotting from multiple sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When plotting from multiple sources, the layout is defined:

.. code-block:: python

    layout = {
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

Notice the different names of the sources. Each name corresponds to the name of the acquisition source, defined in the acquisition class (see `first example <simple_start.html>`_ and `using multiple sources <multiple_sources.html>`_ example).

It is important to note that the subplot locations are the same for all acquisition sources, but the indices of the data are different. 

For example, the subplot at location (0, 0)
will containt the plots from source "DataSource1" with indices 0 and 1, and the plots from source "DataSource2" with indices 0.

Channel vs. channel plot
~~~~~~~~~~~~~~~~~~~~~~~~

When plotting from multiple sources, it is possible to plot the data from one channel of one source against the data from one channel of another source.
Example:

.. code-block:: python

    layout = {
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

.. code-block:: python

    layout = {
        'DataSource': {
            (0, 0): [0, 1],
            (1, 0): [2, 3, function],
        }
    }

The ``function`` can be specified by the user. To use the built-in functions, a string is passed to the ``function`` argument. An example of a built-in function is "fft"
which computes the `Fast Fourier Transform <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html>`_ of the data with indices 2 and 3.

To build a custom function, the function must be defined as follows:

.. code-block:: python

    def function(self, channel_data):
        '''
        :param self: instance of the acquisition object (has to be there so the function is called properly)
        :param channel_data: channel data
        '''
        return channel_data**2

The ``self`` argument in the custom function referes to the instance of the acquisition object. This connection can be used to access the properties of the acquisition object, e.g. sample rate.
The ``channel_data`` argument is a list of numpy arrays, where each array corresponds to the data from one channel. The data is acquired in the order specified in the ``layout`` dictionary.

For the layout example above, the custom function is called for each channel separetely, the ``channel_data`` is a one-dimensional numpy array. To add mutiple channels to the ``channel_data`` argument,
the ``layout`` dictionary is modified as follows:

.. code-block:: python

    layout = {
        'DataSource': {
            (0, 0): [0, 1],
            (1, 0): [(2, 3), function],
        }
    }

The ``function`` is now passed the ``channel_data`` with shape (N, 2) where N is the number of samples.
The function can also return a 2D numpy array with shape (N, 2) where the first column is the x-axis and the second column is the y-axis.
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

.. _subplot_options:
The ``subplot_options``
-----------------------

The properties of each subplot, defined in ``layout`` can be specified with the ``subplot_options`` argument. The ``subplot_options`` argument is a dictionary where the keys are the positions of the subplots.

Example:

.. code-block:: python

    subplot_options = {
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

- ``xlim``: tuple of two floats, the limits of the x-axis
- ``ylim``: tuple of two floats, the limits of the y-axis
- ``axis_style``: string, the style of the axis. Can be "linear", "semilogx", "semilogy" or "loglog"
- ``title``: string, the title of the subplot
- ``rowspan``: int, the number of rows the subplot spans. Default is 1.
- ``colspan``: int, the number of columns the subplot spans. Default is 1.
- ``refresh_rate``: int, the refresh rate of the subplot in milliseconds. If this option is not specified, the refresh rate defined in the :class:`Visualization` is used.

.. note::
    The ``xlim`` defines the samples that are plotted on the x-axis, not only a narrowed view of the data. With this, the same data can be viewed with different zoom levels in an effcient way.

An example of ``subplot_options`` with ``colspan``:

.. code-block:: python

    subplot_options = {
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
