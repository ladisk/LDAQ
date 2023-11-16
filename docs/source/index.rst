.. SDyPy project template documentation master file, created by
   sphinx-quickstart on Wed Jul 10 08:16:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LDAQ documentation!
==============================

What is LDAQ?
-------------

LDAQ stands for **L**\ ightweight **D**\ ata **A**\ c\ **Q**\ uisition, a Python-based toolkit designed to make data collection seamless and efficient. Whether you're a researcher, engineer, or hobbyist, LDAQ offers a powerful yet user-friendly platform to gather 
data from a wide range of hardware sources. The package enables data **acquisition** and signal **generation** from multiple hardware sources, and costumizable **live data visualization**.

Key Features:
-------------

- 🐍 **Python-Powered**: Built on the robust and versatile Python language, LDAQ harnesses its power to offer a streamlined data collection process. It's compatible with all Python environments, ensuring ease of integration into your existing workflows.

- 📟 **Diverse Hardware Compatibility**: LDAQ supports a variety of hardware sources, including:

  - National Instruments

  - Digilent

  - Serial communication devices (i.e. Arduino, ESP)

  - FLIR Cameras

  - Simulated hardware

- 📊 **Advanced Data Visualization & Analysis**: LDAQ doesn’t just collect data; it helps you understand it. With built-in features like real-time signal visualization and Fast Fourier Transform (FFT) analysis, you can dive deep into your data for more insightful discoveries.

- ⚙️ **Customization & Flexibility**: Tailor LDAQ to your specific needs. Whether you're dealing with high-speed data streams or complex signal processing, LDAQ's customizable framework allows you to optimize and accelerate your data acquisition processes.


The source-code for this package is available on `GitHub <https://github.com/ladisk/LDAQ>`_.

See :doc:`simple_start` for a quick start guide.

See :doc:`tutorials` to learn about different LDAQ features.

For more advanced use cases of currently supported hardware see notebook :doc:`examples`.

Table of Contents
=================
.. toctree::
   :maxdepth: 3

   installation
   simple_start
   tutorials
   snippets
   supported_devices
   code
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
