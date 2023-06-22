Code snippets
=============

Use theese code snippets to lightning-fast start your experimental work.

National Instruments
--------------------

Snippets to use with the National Instruments DAQmx driver.

Acquisition only
~~~~~~~~~~~~~~~~

Only the acquisition is used. The data is saved to a file. The visualization is not included.

.. code-block:: python

    import LDAQ

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'

    # Create the acquisition object
    acq = LDAQ.national_instruments.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Create the Core object
    ldaq = LDAQ.Core(acq)

    # Set the trigger on the input task
    ldaq.set_trigger(
        source=input_task_name,
        level=100,
        channel=0,
        duration=11,
        presamples=10
    )

    # Run the acquisition
    ldaq.run()

    # Save the measurment (uncomment when needed)
    # ldaq.save_measurement('filename')


Acquisition with visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single acquisition source is used. The data is visualized in real time. The data is not saved to a file.

Adjust the ``layout`` and ``subplot_options`` to your needs. See `visualization <visualization.html>`_ for details.

.. code-block:: python

    import LDAQ

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'

    # Create the acquisition object
    acq = LDAQ.national_instruments.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Live visualization

    # Create the Visualization object
    vis = LDAQ.Visualization()

    # Add lines
    vis.add_lines((0, 0), source=input_task_name, channels=0)
    vis.add_lines((1, 0), source=input_task_name, channels=1)

    # Edit subplot options
    vis.config_subplot((0, 0), xlim=(0, 1), ylim=(-5, 5), axis_style='linear')
    vis.config_subplot((1, 0), xlim=(0, 1), ylim=(-5, 5), axis_style='linear')


    # Create the Core object
    ldaq = LDAQ.Core(acq, visualization=vis)

    # Set the trigger on the input task
    ldaq.set_trigger(
        source=input_task_name,
        level=100,
        channel=0,
        duration=11,
        presamples=10
    )

    # Run the acquisition
    ldaq.run()

    # Save the measurment (uncomment when needed)
    # ldaq.save_measurement('filename')


Acquisition, generation and visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single acquisition source and a single generation source are used. The data is visualized in real time. The data is not saved to a file.

To use multiple acquisition and generation sources, define them separately and pass them to the :class:`LDAQ.Core` in a list (see the `multiple sources <multiple_sources.html>`_ section).

.. code-block:: python

    import LDAQ
    import pyExSi

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'
    output_task_name = 'OutputTaskName'

    # Create the acquisition object
    acq = LDAQ.national_instruments.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Create the generation object
    # The excitation signal
    time_array = np.arange(100000) / 10000
    signal = np.sin(time_array*2*np.pi*10)
    
    # The generation object
    gen = LDAQ.national_instruments.NIGeneration(output_task_name, signal)

    # Live visualization

    # Create the Visualization object
    vis = LDAQ.Visualization()

    # Add lines
    vis.add_lines((0, 0), source=input_task_name, channels=0)
    vis.add_lines((1, 0), source=input_task_name, channels=1)

    # Edit subplot options
    vis.config_subplot((0, 0), xlim=(0, 1), ylim=(-5, 5), axis_style='linear')
    vis.config_subplot((1, 0), xlim=(0, 1), ylim=(-5, 5), axis_style='linear')


    # Create the Core object
    ldaq = LDAQ.Core(acq, gen, visualization=vis)

    # Set the trigger on the input task
    ldaq.set_trigger(
        source=input_task_name,
        level=100,
        channel=0,
        duration=11,
        presamples=10
    )

    # Run the acquisition
    ldaq.run()

    # Save the measurment (uncomment when needed)
    # ldaq.save_measurement('filename')