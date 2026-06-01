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

    task.add_channel(channel_name='Channel_1', device='cDAQ1Mod1', channel_ind=0,
                     units='V', min_val=-10, max_val=10)
    task.add_channel(channel_name='Channel_2', device='cDAQ1Mod1', channel_ind=1,
                     units='V', min_val=-10, max_val=10)

The ``add_channel`` arguments are:

- ``channel_name``: A unique logical name for the channel.
- ``device``: The NI device or module name string (e.g. ``'Dev1'``, ``'cDAQ1Mod1'``).
- ``channel_ind``: Physical analog-input channel index on the device (e.g. ``0`` for ``ai0``).
- ``units``: Measurement units (e.g. ``'V'``, ``'g'``, ``'N'``). Always required.
- ``min_val`` / ``max_val``: Minimum and maximum expected values in the chosen units.

For sensor channels (accelerometer, force), pass ``sensitivity`` and ``sensitivity_units``
instead of (or in addition to) ``min_val`` / ``max_val`` — see the
`nidaqwrapper documentation <https://github.com/ladisk/nidaqwrapper>`_ for details.

After all channels are added, the task can be passed to the ``NIAcquisition`` object:

.. code:: python

    acq = LDAQ.national_instruments.NIAcquisition(task, acquisition_name='source_name')

For more details, see the `getting started page <simple_start.html>`_.

.. note::

    ``AITask`` and ``AOTask`` are provided by the ``nidaqwrapper`` package and re-exported
    from ``LDAQ.national_instruments`` for convenience. Refer to the
    `nidaqwrapper documentation <https://github.com/ladisk/nidaqwrapper>`_ for the full API
    reference, including TOML-based task configuration.

.. _ni-iepe-channels:

IEPE channels
-------------

The same ``AITask.add_channel`` interface configures IEPE (Integrated
Electronics Piezo-Electric) accelerometers and force sensors — for example on
the NI 9234, NI 9232, or other IEPE-capable modules. Passing ``units='g'`` (or
``'m/s**2'``) together with ``sensitivity`` and ``sensitivity_units`` routes
the channel through ``add_ai_accel_chan``; passing ``units='N'`` routes it
through ``add_ai_force_iepe_chan``. In both cases nidaqmx enables the IEPE
constant-current excitation automatically.

Valid unit strings:

- Acceleration: ``units='g'`` or ``'m/s**2'``; ``sensitivity_units='mV/g'`` or ``'mV/m/s**2'``.
- Force: ``units='N'``; ``sensitivity_units='mV/N'``.

Voltage channels (``units='V'``) can be mixed into the same task — no
``sensitivity`` is needed for those.

A full worked example combining an IEPE impact hammer with two IEPE
accelerometers, with a trigger on the hammer channel for pre- and post-impact
capture, is available in the
:doc:`examples/002_acquisition_NI_IEPE` notebook (typical modal-test setup).

.. _ni-task-output:

Output task
-----------

Output tasks are created with the ``AOTask`` class. First, create an ``AOTask`` object:

.. code:: python

    output_task = LDAQ.national_instruments.AOTask('output_task', sample_rate=25600)

Then add the analog output channels:

.. code:: python

    output_task.add_channel(channel_name='Channel_1', device='cDAQ1Mod2', channel_ind=0,
                            min_val=-10, max_val=10)
    output_task.add_channel(channel_name='Channel_2', device='cDAQ1Mod2', channel_ind=1,
                            min_val=-10, max_val=10)

The ``add_channel`` arguments for output tasks are:

- ``channel_name``: A unique logical name for the channel.
- ``device``: The NI device or module name string (e.g. ``'Dev1'``, ``'cDAQ1Mod2'``).
- ``channel_ind``: AO channel index on the device (e.g. ``0`` for ``ao0``).
- ``min_val`` / ``max_val``: Minimum and maximum output voltage (defaults: ``-10`` and ``10``).

Finally, pass the ``output_task`` to the ``NIGeneration`` class:

.. code:: python

    gen = LDAQ.national_instruments.NIGeneration(output_task, signal, generation_name='source_name')

For more details on the :class:`LDAQ.national_instruments.NIGeneration` class, see the
`generation page <generation.html>`_.

.. note::

    The units of the output channels are Volts.
