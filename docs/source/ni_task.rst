Create NI task
================

In addition to specifying the task in the NI MAX, the task can also be specified in the code. 

Input task
----------

First create a ``NITask`` object:

.. code:: python

    task = LadiskDAQ.NITask('task', sample_rate=1000, settings_file=None)

The arguments of the ``NITask`` class are:

- ``task_name``: The name of the task.
- ``sample_rate``: The sample rate of the task in Hz.
- ``settings_file``: The path to the settings (xmlx) file. Optional.

Then, the channels can be added to the task:

.. code:: python

    task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=0, sensitivity=100, sensitivity_units='mV/g', units='g')
    task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=1, sensitivity=100, sensitivity_units='mV/g', units='g')

After all channels are added to the task, the task can be passed to the ``NIAcquisition`` object:

.. code:: python

    acq = LadiskDAQ.NIAcquisition(task, acquisition_name='source_name')

For more details, see `getting started page <simple_start.html>`_.

.. note::

    The task is not created until the acquisition is started or the ``save`` method is called (see `Save task`_).

Settings file
~~~~~~~~~~~~~

To simplify the creation of the settings file, a settings file can be created. The settings file is a
``xmlx`` file which has the following column names:

- ``serial_nr``: serial number of the sensor.
- ``sensitivity``: sensitivity of the sensor.
- ``sensitivity_units``: units of the sensitivity (see ``LadiskDAQ.UNITS`` for the list of supported units).
- ``units``: units of the sensor.

To use the settings file, pass it to the ``NITask`` object:

.. code:: python

    task = LadiskDAQ.NITask('task', sample_rate=1000, settings_file='settings.xmlx')

Then, when adding the channels, the sensitivity, sensitivity units and units can be ommitted.
The ``channel_name``, ``device_ind`` and ``channel_ind`` are still required. Additionally, the
serial number of the sensor is required to find the correct settings in the settings file.

.. code:: python
    
    task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=0, serial_nr='123')
    task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=1, serial_nr='456')

Custom scale
~~~~~~~~~~~~

It is possible to defina a custom linear scale for the sensor. This can be done by passing the ``scale`` argument
to the ``add_channel`` method (the ``sensitivity`` and ``sensitivity_units`` arguments are then not required):

.. code:: python

    task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=0, units='N', scale=113.2)

The ``scale`` argument must be ``float`` or ``tuple``:

- ``float``: The scale is the slope of the linear function.
- ``tuple``: The first element of the tuple is the slope and the second element is the offset of the linear function.

If the ``scale`` argument is passed, it is assumed, that the measured signal is in ``Volts``. 
The output (scaled) units are specified by the ``units`` argument.

For the example above, the measured signal is in ``Volts`` and the output units are in ``Newtons``.
The scaled units are an arbitrary string and do **not** have to be in the ``LadiskDAQ.UNITS`` list.

Save task
~~~~~~~~~

When the task is created and the channels are added, the task can be saved. The saved task will then 
appear in NI MAX, where it can be edited, deleted, etc.

To save the task, call the ``save`` method of the ``NITask`` object:

.. code:: python

    task.save()

When the task is saved, the ``clear_task()`` method is automatically called. This means that the task cannot be
directly passed to the ``NIAcquisition`` object. In this case the task's name must be passed to the ``NIAcquisition`` (see `getting started page <simple_start.html>`_).

.. note::

    If the user would like to create and save the task and still pass the ``NITask`` object to the ``NIAcquisition`` class directly,
    the following must be called:

    .. code:: python

        task.save(clear_task=False)

Output task
-----------

Output task can also be create by ``LadiskDAQ``. First, create :class:`LadiskDAQ.NITaskOutput` object:

.. code:: python

    output_task = LadiskDAQ.NITaskOutput('task', sample_rate=1000)

Then add the analog output channels:

.. code:: python

    output_task.add_channel(channel_name='Channel_1', device_ind=0, channel_ind=0, min_val=-10, max_val=10)
    output_task.add_channel(channel_name='Channel_2', device_ind=0, channel_ind=1, min_val=-10, max_val=10)

Finally, add the ``output_task`` to the ``NIGenerator`` class (instead of the task name):

.. code:: python

    gen = LadiskDAQ.NIGenerator(output_task, generator_name='source_name')

For more details on :class:`LadiskDAQ.NIGenerator` class, see `generation page <generation.html>`_.

.. note::

    The units of the output channels are ``Volts``.