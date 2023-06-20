import os
import sys
sys.path.insert(0, os.path.realpath('../../'))
import time

import LDAQ


def test_NITask_basic():
    task = LDAQ.national_instruments.NITask('Task', 25600)
    task.add_channel('ch1', 0, 0, units='V', scale=1.)
    task.add_channel('ch2', 0, 1, units='V', scale=1.)

    acq = LDAQ.national_instruments.NIAcquisition(task, 'NI_acq')
    time.sleep(5)

    acq.run_acquisition(1)

    data = acq.get_measurement_dict()

    assert 'time' in data.keys()
    assert 'data' in data.keys()
    assert data['data'].shape == (25600, 2)
    assert 'sample_rate' in data.keys()
    assert data['sample_rate'] == 25600
    assert 'channel_names' in data.keys()
    assert data['channel_names'] == ['ch1', 'ch2']


def test_NITask_core():
    task = LDAQ.national_instruments.NITask('Task', 25600)
    task.add_channel('ch1', 0, 0, units='V', scale=1.)
    task.add_channel('ch2', 0, 1, units='V', scale=1.)

    acq = LDAQ.national_instruments.NIAcquisition(task, 'NI_acq')

    ldaq = LDAQ.Core(acq)
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


def test_NIMAX_basic():
    acq = LDAQ.national_instruments.NIAcquisition("VirtualTask", 'NI_acq')
    time.sleep(5)
    acq.run_acquisition(1)

    data = acq.get_measurement_dict()

    assert 'time' in data.keys()
    assert 'data' in data.keys()
    assert 'sample_rate' in data.keys()
    assert data['sample_rate'] == 25600
    assert 'channel_names' in data.keys()


def test_NIMAX_core():
    acq = LDAQ.national_instruments.NIAcquisition("VirtualTask", 'NI_acq')

    ldaq = LDAQ.Core(acq)
    ldaq.run(1, autostart=True)

    data = ldaq.get_measurement_dict()

    assert 'NI_acq' in data.keys()

    assert 'time' in data['NI_acq'].keys()
    assert 'data' in data['NI_acq'].keys()
    assert 'sample_rate' in data['NI_acq'].keys()
    assert data['NI_acq']['sample_rate'] == 25600
    assert 'channel_names' in data['NI_acq'].keys()
    