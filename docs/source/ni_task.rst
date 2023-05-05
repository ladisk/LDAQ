Create NI task
================

When using National Instrument equipment, the tasks can be specified in the NI MAX software.

The task can also be specified in the code. First create a ``NITask`` object:

.. code:: python

    task = LadiskDAQ.NITask('task_1', sample_rate=1000, settings_file=None)

The arguments of the ``NITask`` class are:

- ``task_name``: The name of the task.
- ``sample_rate``: The sample rate of the task in Hz.
- ``settings_file``: The path to the settings (xmlx) file. Optional.

Then, the channels can be added to the task:

.. code:: python

    task.add_channel('Channel_1', 0, 0, 100, 'mV/g', 'g')
    task.add_channel('Channel_2', 0, 1, 100, 'mV/g', 'g')

After all channels are added to the task, the task can be passed to the ``Core`` object:

.. code:: python

    acq = LadiskDAQ.NIAcquisition(task, acquisition_name='source_name')