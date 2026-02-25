Create NI task
================

In addition to specifying the task in NI MAX, tasks can also be configured programmatically
using the ``AITask`` and ``AOTask`` classes from the `nidaqwrapper <https://github.com/ladisk/nidaqwrapper>`_ package,
which are re-exported from ``LDAQ.national_instruments``.

Input task
----------

First create an ``AITask`` object:

.. code:: python

    task = LDAQ.national_instruments.AITask('task', sample_rate=1000)

The required arguments are:

- ``task_name``: The name of the task.
- ``sample_rate``: The sample rate of the task in Hz.

Then, add the analog input channels to the task:

.. code:: python

    task.add_channel(device_name='cDAQ1Mod1', channel='ai0', channel_name='Channel_1',
                     terminal_config='DEFAULT', voltage_range=(-10, 10))
    task.add_channel(device_name='cDAQ1Mod1', channel='ai1', channel_name='Channel_2',
                     terminal_config='DEFAULT', voltage_range=(-10, 10))

The ``add_channel`` arguments are:

- ``device_name``: The NI device or module name (e.g. ``'Dev1'``, ``'cDAQ1Mod1'``).
- ``channel``: The channel string (e.g. ``'ai0'``, ``'ai1'``).
- ``channel_name``: A human-readable name for the channel.
- ``terminal_config``: Terminal configuration (e.g. ``'DEFAULT'``, ``'RSE'``, ``'NRSE'``, ``'DIFF'``).
- ``voltage_range``: A tuple ``(min_val, max_val)`` specifying the input voltage range in Volts.

After all channels are added, the task can be passed to the ``NIAcquisition`` object:

.. code:: python

    acq = LDAQ.national_instruments.NIAcquisition(task, acquisition_name='source_name')

For more details, see the `getting started page <simple_start.html>`_.

.. note::

    ``AITask`` and ``AOTask`` are provided by the ``nidaqwrapper`` package and re-exported
    from ``LDAQ.national_instruments`` for convenience. Refer to the
    `nidaqwrapper documentation <https://github.com/ladisk/nidaqwrapper>`_ for the full API
    reference, including TOML-based task configuration.

.. _ni-task-output:

Output task
-----------

Output tasks are created with the ``AOTask`` class. First, create an ``AOTask`` object:

.. code:: python

    output_task = LDAQ.national_instruments.AOTask('output_task', sample_rate=25600)

Then add the analog output channels:

.. code:: python

    output_task.add_channel(device_name='cDAQ1Mod2', channel='ao0', channel_name='Channel_1',
                            voltage_range=(-10, 10))
    output_task.add_channel(device_name='cDAQ1Mod2', channel='ao1', channel_name='Channel_2',
                            voltage_range=(-10, 10))

The ``add_channel`` arguments for output tasks are:

- ``device_name``: The NI device or module name (e.g. ``'Dev1'``, ``'cDAQ1Mod2'``).
- ``channel``: The channel string (e.g. ``'ao0'``, ``'ao1'``).
- ``channel_name``: A human-readable name for the channel.
- ``voltage_range``: A tuple ``(min_val, max_val)`` specifying the output voltage range in Volts.

Finally, pass the ``output_task`` to the ``NIGeneration`` class:

.. code:: python

    gen = LDAQ.national_instruments.NIGeneration(output_task, signal, generation_name='source_name')

For more details on the :class:`LDAQ.national_instruments.NIGeneration` class, see the
`generation page <generation.html>`_.

.. note::

    The units of the output channels are Volts.
