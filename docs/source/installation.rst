Installation
============

To install LDAQ first create a new environment:

.. code:: bash

    py -m -3.11 venv <environment_name>

.. note::

    This package works with Python versions 3.8 and above.

Then activate the environment:

.. code:: bash

    <environment_name>\Scripts\activate

When in the environemnt, install the package from `GitHub <https://github.com/ladisk/LDAQ>`_
(you will have to have `git <https://git-scm.com/downloads>`_ installed):

.. code:: bash

    pip install git+https://github.com/ladisk/LDAQ

.. note::

    For some functionalities to work, you will need additional packages installed, e.g., for using Flir.


.. note::

    To use **National Instruments** DAQ, you will need to install the `NI-DAQmx <https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#346210>`_ driver.

