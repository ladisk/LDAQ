Code snippets
=============

Use theese code snippets to lightning-fast start your experimental work.

National Instruments
-------------------

Snippets to use with the National Instruments DAQmx driver.

Acquisition only
~~~~~~~~~~~~~~~

Only the acquisition is used. The data is saved to a file. The visualization is not included.

.. code-block:: python

    import LadiskDAQ

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'

    # Create the acquisition object
    acq = LadiskDAQ.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Create the Core object
    ldaq = LadiskDAQ.Core(acq)

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

    import LadiskDAQ

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'

    # Create the acquisition object
    acq = LadiskDAQ.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Live visualization
    # Define the layout (adjust to your needs)
    layout = {
        input_task_name: {
            (0, 0): [0],
            (1, 0): [0],
        }
    }

    # Define the subplot options (adjust to your needs)
    subplot_options = {
        (0, 0): {
            'xlim': (0, 1),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        },
        (1, 0): {
            'xlim': (0, 1),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        }
    }

    # Create the Visualization object
    vis = LadiskDAQ.Visualization(layout, subplot_options, nth="auto")


    # Create the Core object
    ldaq = LadiskDAQ.Core(acq, visualization=vis)

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

To use multiple acquisition and generation sources, define them separately and pass them to the :class:`LadiskDAQ.Core` in a list (see the `multiple sources <multiple_sources.html>`_ section).

.. code-block:: python

    import LadiskDAQ
    import pyExSi

    # Define the task name, specified in NI MAX
    input_task_name = 'TaskName'
    output_task_name = 'OutputTaskName'

    # Create the acquisition object
    acq = LadiskDAQ.NIAcquisition(input_task_name, acquisition_name=input_task_name)

    # Create the generation object
    # The excitation signal
    time_array = np.arange(100000) / 10000
    signal = np.sin(time_array*2*np.pi*10)
    
    # The generation object
    gen = LadiskDAQ.NIGenerator(output_task_name, signal)

    # Live visualization
    # Define the layout (adjust to your needs)
    layout = {
        input_task_name: {
            (0, 0): [0],
            (1, 0): [0],
        }
    }

    # Define the subplot options (adjust to your needs)
    subplot_options = {
        (0, 0): {
            'xlim': (0, 1),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        },
        (1, 0): {
            'xlim': (0, 1),
            'ylim': (-5, 5),
            'axis_style': 'linear'
        }
    }

    # Create the Visualization object
    vis = LadiskDAQ.Visualization(layout, subplot_options, nth="auto")


    # Create the Core object
    ldaq = LadiskDAQ.Core(acq, gen, visualization=vis)

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