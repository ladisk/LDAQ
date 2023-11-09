Virtual Channels
================

For any acquisition source, it is possible to create new virtual channels that are based on the data of physical channels. They can be created
by using ``add_virtual_channel`` method of acquisition source:

.. code-block:: python

    # define a function that will create a new channel:
    def func(ch1, ch2): 
        return ch1 / ch2 # ensure that shape is (N, 1) and not (N, )
    
    # assume 'acq' is our acquisition source
    acq.add_virtual_channel(virtual_channel_name="name_of_new_channel",
                            source_channels=["name_of_channel_1", "name_of_channel_2"],
                            function=func)


.. note::

    Number of arguments of ``func`` has to be the same as number elements in ``source_channels``.

The acquisition source will now contain a new channel named ``name_of_new_channel``. It is equivalent to any physical channels and can be used in visualizations and is stored when measurement is saved. 
For more information see the documentation of ``add_virtual_channel`` method and :doc:`examples`.