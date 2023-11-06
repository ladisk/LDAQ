Multiple sources
================

In the :class:`LDAQ.Core`, multiple sources can be added.

Multiple acquisition sources
----------------------------

To add multiple acquisition sources, the sources are defined separetely.

The following is an example, where two sources are defined, one for a NI DAQ and one for an Arduino.

.. code-block:: python

    acq1 = LDAQ.national_instruments.NIAcquisition('VirtualTask', acquisition_name="NI_task")

    acq2 = LDAQ.serial_communication.SerialAcquisition(port="COM6", baudrate=250000, timeout=5,
                                  byte_sequence=(("int16", 1), ("int16", 1)),
                                  start_bytes=b"\xfa\xfb",
                                  end_bytes=b"\n",
                                  pretest_time= None,
                                  sample_rate=500.,
                                  acquisition_name="Arduino"
                                  )


The sources can simply be added to the :class:`LDAQ.Core` object:

.. code-block:: python

    ldaq = LDAQ.Core(acquisitions=[acq1, acq2], visualization=vis)

``vis`` is the visualization object (for more details see `visualization <visualization.html>`_).

.. note::
    
    The sample rates of the acquisition sources are not always known. If the ``sample_rate`` argument is set to ``None``, the sample rate detection routine is started and
    sample rate is automatically estimated.

.. note::

    Currently, data obtained from multiple acquisition sources is not synchronized. The delay between the sources depends on various factors, mainly the sample rate and 
    hardware data buffer.

Multiple generation sources
---------------------------

Similarly, multiple generation sources can be added. Each generation source is defined separately (see `generation <generation.html>`_) and added to the :class:`LDAQ.Core` object.

.. code-block:: python

    gen1 = ...

    gen2 = ...

    ldaq = LDAQ.Core(acquisition=[acq1, acq2], generations=[gen1, gen2], visualization=vis)