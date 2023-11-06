Installation
============

1) To install LDAQ first create a new environment:

.. code:: bash

    py -m -3.11 venv <environment_name>

The `3.11` refers to the python version you are using. You can check your version by typing:

.. code:: bash

    py --version

.. note::

    This package works with Python versions 3.8 and above.

2) Then activate the environment:

.. code:: bash

    <path_to_venv>\Scripts\activate

3) When the `venv` environment is activated, install the package from `GitHub <https://github.com/ladisk/LDAQ>`_
(you will have to have `git <https://git-scm.com/downloads>`_ installed):

.. code:: bash

    pip install git+https://github.com/ladisk/LDAQ

.. note::

    For some hardware additional drivers are packages are required. See the :doc:`code` for more information.

.. note::

    For example to use **National Instruments** DAQ, you will need to install the `NI-DAQmx <https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#346210>`_ driver.

