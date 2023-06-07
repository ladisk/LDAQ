import os
import sys
sys.path.insert(0, os.path.realpath('../'))

import LadiskDAQ


def test_NITask_basic():
    task = LadiskDAQ.national_instruments.NITask('Task', 25600)
    task.add_channel('ch1', 0, 0, units='V', scale=1.)
    task.add_channel('ch2', 0, 1, units='V', scale=1.)

    acq = LadiskDAQ.national_instruments.NIAcquisition(task, 'NI_acq')

    acq.run_acquisition(1)

    data = acq.get_measurement_dict()

    assert 'time' in data.keys()
    assert 'data' in data.keys()
    assert data['data'].shape == (25600, 2)
    assert 'sample_rate' in data.keys()
    assert data['sample_rate'] == 25600
    assert 'channel_names' in data.keys()
    assert data['channel_names'] == ['ch1', 'ch2']


def test_NITask_basic_2():
    task = LadiskDAQ.national_instruments.NITask('Task', 25600)
    task.add_channel('ch1', 0, 0, units='V', scale=1.)
    task.add_channel('ch2', 0, 1, units='V', scale=1.)

    acq = LadiskDAQ.national_instruments.NIAcquisition(task, 'NI_acq')

    ldaq = LadiskDAQ.Core(acq)
    ldaq.run(1, autostart=True)

    data = ldaq.get_measurement_dict()

    assert 'NI_acq' in data.keys()

    assert 'time' in data['NI_acq'].keys()
    assert 'data' in data['NI_acq'].keys()
    assert data['NI_acq']['data'].shape == (25600, 2)
    assert 'sample_rate' in data['NI_acq'].keys()
    assert data['NI_acq']['sample_rate'] == 25600
    assert 'channel_names' in data['NI_acq'].keys()
    assert data['NI_acq']['channel_names'] == ['ch1', 'ch2']


def test_Visualization_basic():
    vis = LadiskDAQ.Visualization()
    vis.add_lines((0, 0), source='NI_acq', channels=['ch1', 'ch2'])
    vis.add_lines((0, 1), source='NI_acq', channels=['ch1', 1])

    vis.add_image('NI_acq', 'image1')
    vis.add_image('NI_acq', 1)

    assert 'NI_acq' in vis.plots.keys()
    assert vis.plots['NI_acq'][0]['pos'] == (0, 0)
    assert vis.plots['NI_acq'][1]['pos'] == (0, 0)
    assert vis.plots['NI_acq'][2]['pos'] == (0, 1)
    assert vis.plots['NI_acq'][3]['pos'] == (0, 1)

    assert vis.plots['NI_acq'][0]['channels'] == 'ch1'
    assert vis.plots['NI_acq'][1]['channels'] == 'ch2'
    assert vis.plots['NI_acq'][2]['channels'] == 'ch1'
    assert vis.plots['NI_acq'][3]['channels'] == 1


def test_Visualization_basic_2():
    vis = LadiskDAQ.Visualization()
    vis.add_lines((0, 0), source='NI_acq', channels=['ch1', 'ch2'])
    vis.add_lines((0, 1), source='NI_acq', channels=['ch1', 1])

    vis.config_subplot((0, 0), xlim=(0, 10), ylim=(0, 10), title='Test title', axis_style='semilogy')

    assert (0, 0) in vis.subplot_options.keys()
    assert vis.subplot_options[(0, 0)]['xlim'] == (0, 10)
    assert vis.subplot_options[(0, 0)]['ylim'] == (0, 10)
    assert vis.subplot_options[(0, 0)]['title'] == 'Test title'
    assert vis.subplot_options[(0, 0)]['axis_style'] == 'semilogy'
    assert vis.subplot_options[(0, 0)]['colspan'] == 1
    assert vis.subplot_options[(0, 0)]['rowspan'] == 1