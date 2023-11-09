Custom Acquisitions
===================

General Guidelines
------------------

It is possible to create custom acquisition classes that can be used with the LDAQ library. This is done by subclassing the :class:`LDAQ.acquisition_base.BaseAcquisition` class and implementing
the abstract methods. The following is a list of guidelines that should be followed when creating custom acquisition classes:

.. autoclass:: LDAQ.acquisition_base.BaseAcquisition
    :members: read_data, terminate_data_source, clear_buffer, get_sample_rate
    :noindex:


Example
-------

In the example below, the source code of National Instrument acquisition class is shown.

.. literalinclude:: ../../LDAQ/national_instruments/acquisition.py
    :language: python
    :linenos:
    :caption: Source code of National Instrument acquisition class


