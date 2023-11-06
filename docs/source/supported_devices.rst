Supported Devices
=================

Acqusition
----------

- :class:`LDAQ.national_instruments.NIAcquisition`: Any National Instruments input device.

- :class:`LDAQ.serial_communication.SerialAcquisition`: Any device that communicates via serial port, tested with Arduino and ESP32.

- :class:`LDAQ.digilent.WaveFormsAcquisition`: Any Digilent device, tested with Analog Discovery 2.

- :class:`LDAQ.flir.FLIRThermalCamera`: Based on pySpin, tested with FLIR A50.

- :class:`LDAQ.basler.BaslerCamera`: Based on pyPylon, any Basler Camera should work.

- :class:`LDAQ.simulator.SimulatedAcquisition`: Used to test the GUI without any hardware connected. Alternatively it can be used to track memory, CPU and disk usage.


Generation
----------

- :class:`LDAQ.national_instruments.NIGeneration`: Any National Instruments output device.

Hardware-Specific Installation Instructions
-------------------------------------------

.. autodata:: LDAQ.national_instruments.NIAcquisition
    :annotation:
    :noindex:

.. autodata:: LDAQ.flir.FLIRThermalCamera
    :annotation:
    :noindex:

.. autodata:: LDAQ.digilent.WaveFormsAcquisition
    :annotation:
    :noindex:

.. autodata:: LDAQ.basler.BaslerCamera
    :annotation:
    :noindex:



