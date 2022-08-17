Basic usage is presented here. For further details, see the `documentation <https://ladiskdaq.readthedocs.io/en/latest/getting_started.html>`_.

Basic usage
===========
Define the acquisition object and set the trigger:
::

    acq = LadiskDAQ.NIAcquisition(input_task_name)
    acq.set_trigger(level=100, channel=0, duration=11, presamples=10)

Optionally, the generation object can also be created:
::

    gen = LadiskDAQ.NIGenerator(output_task_name, signal)

Combine ``acq`` and ``gen`` in the ``LDAQ`` class (``gen`` is optional):
::

    ldaq = LadiskDAQ.LDAQ(acq, gen)

Customize the plot:
::

    ldaq.configure(plot_layout='default', max_time=5.0, nth_point='auto', autoclose=True, refresh_interval=0.01)

To start the acquisition and visualization, call:
::

    ldaq.run()