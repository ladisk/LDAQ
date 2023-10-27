Code documentation
==================

Overview
--------

This section provides comprehensive documentation for the LDAQ code, designed to facilitate data acquisition, generation, and visualization across a variety of hardware platforms. The documentation is organized into several key sections, each focusing on a specific aspect of the LDAQ system.

The "Acquisition" section introduces the base acquisition class along with hardware-specific acquisition classes, detailing the methods and properties associated with data acquisition from various devices. These devices range from simulated environments and serial communications to specific hardware integrations like National Instruments, Digilent, FLIR thermal cameras, and Basler cameras.

In the "Generation" section, the focus shifts to data generation. Here, the base generation class and various hardware-specific generation classes are documented, providing insights into how data can be generated and manipulated within the LDAQ framework.

The "Visualization" section delves into the capabilities of LDAQ in terms of data representation and user interface, highlighting the various features of the LDAQ Visualization class.

Under "Core," the core functionalities of LDAQ are encapsulated, documenting the central operations and workflows that tie together acquisition, generation, and visualization.

The "Utilities" section encompasses additional tools and functions that aid in the operation of LDAQ, including National Instruments utilities and other general-purpose functions for loading and handling measurement data.

Acquisition
-----------

Base Acqusition Class
~~~~~~~~~~~~~~~~~~~~~~

All acquisition classes inherit from the base class :class:`LDAQ.acquisition_base.BaseAcquisition`. This class defines the basic interface for all acquisition classes. The following methods are defined:

.. autoclass:: LDAQ.acquisition_base.BaseAcquisition
    :members:

Hardware-Specific Acqusition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LDAQ.simulator.SimulatedAcquisition
    :members:

.. autoclass:: LDAQ.national_instruments.NIAcquisition
    :members:

.. autoclass:: LDAQ.serial_communication.SerialAcquisition
    :members:

.. autoclass:: LDAQ.digilent.WaveFormsAcquisition
    :members:

.. autoclass:: LDAQ.flir.FLIRThermalCamera
    :members:

.. autoclass:: LDAQ.basler.BaslerCamera
    :members:

Generation
----------

Base Generation Class
~~~~~~~~~~~~~~~~~~~~~

All generation classes inherit from the base class :class:`LDAQ.generation_base.BaseGeneration`. This class defines the basic interface for all generation classes. The following methods are defined:

.. autoclass:: LDAQ.generation_base.BaseGeneration
    :members:

Hardware-Specific Generation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LDAQ.national_instruments.NIGeneration
    :members:

Visualization
-------------

.. autoclass:: LDAQ.Visualization
    :members:

Core
-----

.. autoclass:: LDAQ.core.Core
    :members:

Utilities
---------

National Instruments
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LDAQ.national_instruments.NITask
    :members:

.. autoclass:: LDAQ.national_instruments.NITaskOutput
    :members:

Other
~~~~~

.. autofunction:: LDAQ.utils.load_measurement

.. autofunction:: LDAQ.load_measurement_multiple_files