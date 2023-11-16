LDAQ - Streamlined Data Acquisition and Generation
==================================================

What is LDAQ?
-------------

LDAQ stands for **L**ightweight **D**ata **A**cquisition, a Python-based toolkit designed to make data collection seamless and efficient. Whether you're a researcher, engineer, or hobbyist, LDAQ offers a powerful yet user-friendly platform to gather 
data from a wide range of hardware sources.

.. image:: /docs/source/images/frf_visualization.gif
   :alt: frf_visualization

Key Features:
-------------

- 🐍 **Python-Powered**: Built on the robust and versatile Python language, LDAQ harnesses its power to offer a streamlined data collection process. It's compatible with all Python environments, ensuring ease of integration into your existing workflows.

- 🔄 **Diverse Hardware Compatibility**: LDAQ supports a variety of hardware sources, including:
  - National Instruments
  - Digilent
  - Serial communication devices (i.e. Arduino, ESP)
  - FLIR Cameras

  This wide range of compatibility makes LDAQ a versatile tool for various data acquisition needs. Also includes simulated hardware for workflow developments, and ability to include currently unsupported hardware. 

- 📊 **Advanced Data Visualization & Analysis**: LDAQ doesn’t just collect data; it helps you understand it. With built-in features like real-time signal visualization and Fast Fourier Transform (FFT) analysis, you can dive deep into your data for more insightful discoveries.

- ⚙️ **Customization & Flexibility**: Tailor LDAQ to your specific needs. Whether you're dealing with high-speed data streams or complex signal processing, LDAQ's customizable framework allows you to optimize and accelerate your data acquisition processes.


Getting started
===============

Dive into the world of efficient data acquisition with LDAQ. Our `documentation <https://ldaq.readthedocs.io/en/latest>`_ will guide you through installation, setup, and basic usage to get you up and running in no time.

.. image:: /docs/source/images/getting_started.gif
   :alt: demo_gif

Installation
------------

The package can be installed from PyPI using pip:
.. code-block::

    pip install LDAQ


Create the acquisition object
-----------------------------

The first step to starting the measurement is to create an acquisition object. Depending on your measurement hardware,
you can select the appropriate acquisition class. 

In this example, we use the ``LDAQ.national_instruments.NIAcquisition`` class, which is
a wrapper for the National Instruments DAQmx driver. The class accepts the name of the input task as an argument:

.. code-block:: python

    acq = LDAQ.national_instruments.NIAcquisition(input_task_name, acquisition_name='DataSource')

If the  ``acquisition_name`` argument is not specified, the name of the acquisition object will be set to the value of ``input_task_name``.

The ``acquisition_name`` argument is important when using multiple acquisition objects in the same measurement, and when specifying the layout of the
live `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_.

Create the ``Core`` object
--------------------------

The ``acq`` object can now be added to the ``LDAQ.Core`` class:

.. code-block:: python

    ldaq = LDAQ.Core(acq)

.. note::

    To add live visualization of the measurement, the visualization object can be added to the ``LDAQ.Core`` object:

    .. code-block:: python

        ldaq = LDAQ.Core(acq, visualization=vis)

    Read how to prepare the ``vis`` object in the `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_ section.

Set the trigger
---------------

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

    The ``LDAQ.Core`` may seem unnecessary when using a single acquisition source.
    However, it enables the simultaneous usage of signal `generation <https://ldaq.readthedocs.io/en/latest/generation.html>`_, live `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_ 
    and `multiple acquisition/generation <https://ldaq.readthedocs.io/en/latest/multiple_sources.html>`_ sources.

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

- Add generation to the ``LDAQ.Core`` object (see `generation <https://ldaq.readthedocs.io/en/latest/generation.html>`_).
- Apply virtual channels to acquisition objects, to perform calculations on the acquired data (see `virtual channels <https://ldaq.readthedocs.io/en/latest/virtual_channels.html>`_).
- Add visualization to the ``LDAQ.Core`` object (see `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_).
- Apply functions to measured data in real-time visualization (see `visualization <https://ldaq.readthedocs.io/en/latest/visualization.html>`_).
- Add multiple acquisition and signal generation objects to ``LDAQ.Core`` (see `multiple sources <https://ldaq.readthedocs.io/en/latest/multiple_sources.html>`_).
- Define a NI Task in your program and use it with ``LDAQ`` (see `NI Task <https://ldaq.readthedocs.io/en/latest/ni_task.html>`_).
- Currently the package supports a limited set of devices from National Instruments, Digilent, FLIR, Basler and devices using serial communication (see `supported devices <https://ldaq.readthedocs.io/en/latest/supported_devices.html>`_).
- Create your own acquisition class by overriding just few methods (see `custom acquisition <https://ldaq.readthedocs.io/en/latest/custom_acquisitions_and_generations.html>`_).