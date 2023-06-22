Basic usage is presented here. For further details, see the `documentation <https://ldaq.readthedocs.io/en/latest/index.html>`_.

Basic usage
===========

Create the acquisition object
-----------------------------

The first step to starting the measurement is to create an acquisition object. Depending on your measurement hardware,
you can select the appropriate acquisition class. 

In this example, we use the ``LDAQ.national_instrument.NIAcquisition`` class, which is
a wrapper for the National Instruments DAQmx driver. The class accepts the name of the input task (created in NI MAX) as an argument:

.. code-block:: python

    acq = LDAQ.national_instrument.NIAcquisition(input_task_name, acquisition_name='NI_data_source')

The ``acquisition_name`` argument is important when using multiple acquisition objects in the same measurement, and when specifying the layout of the
live `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_.

Create the ``Core`` object
--------------------------

The ``acq`` object can now be added to the ``LDAQ.Core`` class, which handles the acquisition and generation of signals, and the live visualization of the measurement:

.. code-block:: python

    ldaq = LDAQ.Core(acquisitions=[acq])

.. note::

    To add live visualization of the measurement, the visualization ``LDAQ.Visualization`` object can be added to the ``LDAQ.Core`` object:

    .. code-block:: python

        ldaq = LDAQ.Core(acquisitions=[acq], visualization=vis)

    Read how to prepare the ``vis`` object in the `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_ section.

Set the trigger
---------------

Often the measurement is started when one of the signal excedes a certain level. This can be achieved by setting the trigger on one of the data sources by calling the ``set_trigger`` method:

.. code-block:: python
    
    ldaq.set_trigger(
        source='NI_data_source',
        level=100,
        channel=0, 
        duration=11, 
        presamples=10
    )

Where:

- ``source``: the name of the acquisition object on which the trigger is set.
- ``level``: the trigger level.
- ``channel``: the channel on which the trigger is set.
- ``duration``: the duration of the trigger in seconds.
- ``presamples``: the number of samples to be acquired before the trigger is detected.

.. note::

    The ``LDAQ.Core`` may seem unnecessary when using a single acquisition source.
    However, it enables the usage of signal `generation <https://ldaq.readthedocs.io/en/latest/generation.html>`_, live `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_ and `multiple acquisition/generation <https://ldaq.readthedocs.io/en/latest/multiple_sources.html>`_ sources.

Run the measurement
-------------------

The measurement can now be started by calling the ``run`` method:

.. code-block:: python

    ldaq.run()

Save the measurement
--------------------

After the measurement is completed, the data can be saved by calling:

.. code-block:: python

    ldaq.save_measurement(
        name='my_measurement',
        root=path_to_save_folder,
        timestamp=True,
        comment='my comment'
    )

Where:

- ``name``: required, the name of the measurement, without extension (``.pkl`` is added automatically).
- ``root``: optional, the path to the folder where the measurement will be saved. If it is not given, the measurement will be saved in the current working directory.
- ``timestamp``: optional, add a timestamp at the beginning of the file name.
- ``comment``: optional, a comment to be saved with the measurement.

What else can I do with LDAQ?
-----------------------------

- Add signal generation to the ``LDAQ.Core`` object. (see `generation <https://ldaq.readthedocs.io/en/latest/generation.html>`_)
- Add visualization to the ``LDAQ.Core`` object. (see visualization_)
- Apply functions to measured data in real-time visualization. (see visualization_)
- Apply virtual channels to acquisition objects, to perform calculations on the acquired data. (see `virtual channels <https://ldaq.readthedocs.io/en/latest/virtual_channels.html>`_)
- Add multiple acquisition and signal generation objects to ``LDAQ.Core``. (see `multiple sources <https://ldaq.readthedocs.io/en/latest/multiple_sources.html>`_)
- Define a NI Task in your program and use it with LDAQ. (see `NI Task <https://ldaq.readthedocs.io/en/latest/ni_task.html>`_)
- Create your own acquisition class by overriding just few methods. LDAQ support signal as well as video acquisition sources. (see `custom acquisition <https://ldaq.readthedocs.io/en/latest/custom_acquisition.html>`_)
- Currently the package supports a limited set of devices from National Instruments, Digilent, FLIR, Basler and devices using serial communication. (see `supported devices <https://ldaq.readthedocs.io/en/latest/supported_devices.html>`_)

.. _visualization: https://ldaq.readthedocs.io/en/latest/visualization.html