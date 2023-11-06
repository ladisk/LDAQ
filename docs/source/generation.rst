Generation
==========

The signal generation can also be added to the :class:`LDAQ.Core` class.
Here, an example using the National Instruments DAQmx driver is shown. 

The signal must be prepared by the user (here we recommend to check `pyExSi <https://github.com/ladisk/pyExSi>`_).

.. code-block:: python

    import numpy as np

    fs = 25600 # output sample rate
    t = np.arange(fs * 10) / fs  
    signal1 = np.sin(2*np.pi*800*t) 
    signal2 = np.sin(2*np.pi*450*t) 

    signal = np.array([signal1, signal2]).T # shape must be (n_samples, n_channels)


Signal array is then passed to the :class:`LDAQ.NIGenerator` class. The ``output_task_name`` must be defined beforehand using
`NIMax <https://www.ni.com/en/support/documentation/supplemental/21/what-is-ni-measurement---automation-explorer--ni-max--.html>`_ or
:class:`NITaskOutput` (see :ref:`ni-task-output`).

.. code-block:: python

    gen = LDAQ.national_instruments.NIGeneration(output_task_name, signal)

The ``gen`` object can then be added to the :class:`LDAQ.Core` class:

.. code-block:: python

    ldaq = LDAQ.Core(acq, gen)

``acq`` is the acquisition object, see `first example <simple_start.html>`_.

When the ``.run()`` method is called, the signal is generated and the acquisition is started.

.. note::

    For generating the signal on multiple channels, the ``signal`` must be a numpy array with the shape ``(n_samples, n_channels)``.