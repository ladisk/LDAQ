Generation
==========

The signal generation can also be added to the :class:`LadiskDAQ.Core` class.
Here, an example using the National Instruments DAQmx driver is shown. The signal must be prepared by the user (`pyExSi <https://github.com/ladisk/pyExSi>`_ can be used)
and is passed to the :class:`LadiskDAQ.NIGenerator` class. The ``output_task_name`` must be defined.

.. code-block:: python

    gen = LadiskDAQ.NIGenerator(output_task_name, signal)

The ``gen`` object can then be added to the :class:`LadiskDAQ.Core` class:

.. code-block:: python

    ldaq = LadiskDAQ.Core(acq, gen)

``acq`` is the acquisition object, see `first example <simple_start.html>`_.

When the ``.run()`` method is called, the signal is generated and the acquisition is started.