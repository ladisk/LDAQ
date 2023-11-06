Acquisition
===========

The signal acquisition is first created using any child acquisition class of :class:`LDAQ.BaseAcquisition`. 

In this example, we use the :class:`LDAQ.national_instruments.NIAcquisition` class, which is
a wrapper for the National Instruments DAQmx driver. The class accepts the name of the input task as an argument:

.. code-block:: python

    acq = LDAQ.national_instruments.NIAcquisition(task_name='input_task_1', acquisition_name='NI')

``task_name`` is the input task name defined in NIMax software, and ``acquisition_name`` is the name of acquisition object. The ``acquisition_name`` argument is important when using multiple acquisition objects in the same measurement, and when specifying the layout of the
live `visualization <visualization.html>`_. In the next step the ``acq`` object can be added to the :class:`LDAQ.Core` class:

.. code-block:: python

    ldaq = LDAQ.Core(acq)

Often the measurement is started when one of the signal excedes a certain level. This can be achieved by setting the trigger on one of the data sources by calling the ``set_trigger`` method:

.. code-block:: python
    
    ldaq.set_trigger(
        source='DataSource',
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

    The :class:`LDAQ.Core` may seem unnecessary when using a single acquisition source.
    However, it enables the simultaneous usage of signal `generation <generation.html>`_, live `visualization <visualization.html>`_ and `multiple acquisition/generation <multiple_sources.html>`_ sources.


The measurement can now be started by calling the ``run`` method:

.. code-block:: python

    ldaq.run()

.. note::

    When calling ``run`` method as shown above, the measurement duration will be equal to the one specified with ``set_trigger`` method. 
    Alternatively, the measurement duration can be specified by calling ``run`` method with the ``measurement_duration`` argument:

    .. code-block:: python

        ldaq.run(measurement_duration=10)

After the measurement is completed, the data can be saved by calling ``save_measurmeent`` method:

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
