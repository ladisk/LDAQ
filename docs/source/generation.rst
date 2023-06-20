Generation
==========

The signal generation can also be added to the :class:`LDAQ.Core` class.
Here, an example using the National Instruments DAQmx driver is shown. 

The signal must be prepared by the user (`pyExSi <https://github.com/ladisk/pyExSi>`_ can be used)
and is passed to the :class:`LDAQ.NIGenerator` class. The ``output_task_name`` must be defined.

.. code-block:: python

    gen = LDAQ.national_instruments.NIGeneration(output_task_name, signal)

The ``gen`` object can then be added to the :class:`LDAQ.Core` class:

.. code-block:: python

    ldaq = LDAQ.Core(acq, gen)

``acq`` is the acquisition object, see `first example <simple_start.html>`_.

When the ``.run()`` method is called, the signal is generated and the acquisition is started.

.. note::

    For generating the signal on multiple channels, the ``signal`` must be a numpy array with the shape ``(n_samples, n_channels)``.