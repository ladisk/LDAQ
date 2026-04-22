import os
import sys
sys.path.insert(0, os.path.realpath('../../'))
import time

import LDAQ


# def test_AITask_basic():
#     task = LDAQ.national_instruments.AITask('Task', sample_rate=25600)
#     task.add_channel(device_name='Dev1', channel='ai0', channel_name='ch1', voltage_range=(-10, 10))
#     task.add_channel(device_name='Dev1', channel='ai1', channel_name='ch2', voltage_range=(-10, 10))

#     acq = LDAQ.national_instruments.NIAcquisition(task, acquisition_name='NI_acq')
#     time.sleep(5)

#     acq.run_acquisition(1)

#     data = acq.get_measurement_dict()

#     assert 'time' in data.keys()
#     assert 'data' in data.keys()
#     assert data['data'].shape == (25600, 2)
#     assert 'sample_rate' in data.keys()
#     assert data['sample_rate'] == 25600
#     assert 'channel_names' in data.keys()
#     assert data['channel_names'] == ['ch1', 'ch2']


# def test_AITask_core():
#     task = LDAQ.national_instruments.AITask('Task', sample_rate=25600)
#     task.add_channel(device_name='Dev1', channel='ai0', channel_name='ch1', voltage_range=(-10, 10))
#     task.add_channel(device_name='Dev1', channel='ai1', channel_name='ch2', voltage_range=(-10, 10))

#     acq = LDAQ.national_instruments.NIAcquisition(task, acquisition_name='NI_acq')

#     ldaq = LDAQ.Core(acq)
#     ldaq.run(1, autostart=True)

#     data = ldaq.get_measurement_dict()

#     assert 'NI_acq' in data.keys()

#     assert 'time' in data['NI_acq'].keys()
#     assert 'data' in data['NI_acq'].keys()
#     assert data['NI_acq']['data'].shape == (25600, 2)
#     assert 'sample_rate' in data['NI_acq'].keys()
#     assert data['NI_acq']['sample_rate'] == 25600
#     assert 'channel_names' in data['NI_acq'].keys()
#     assert data['NI_acq']['channel_names'] == ['ch1', 'ch2']


# def test_NIMAX_basic():
#     acq = LDAQ.national_instruments.NIAcquisition("VirtualTask", acquisition_name='NI_acq')
#     time.sleep(5)
#     acq.run_acquisition(1)

#     data = acq.get_measurement_dict()

#     assert 'time' in data.keys()
#     assert 'data' in data.keys()
#     assert 'sample_rate' in data.keys()
#     assert data['sample_rate'] == 25600
#     assert 'channel_names' in data.keys()


# def test_NIMAX_core():
#     acq = LDAQ.national_instruments.NIAcquisition("VirtualTask", acquisition_name='NI_acq')

#     ldaq = LDAQ.Core(acq)
#     ldaq.run(1, autostart=True)

#     data = ldaq.get_measurement_dict()

#     assert 'NI_acq' in data.keys()

#     assert 'time' in data['NI_acq'].keys()
#     assert 'data' in data['NI_acq'].keys()
#     assert 'sample_rate' in data['NI_acq'].keys()
#     assert data['NI_acq']['sample_rate'] == 25600
#     assert 'channel_names' in data['NI_acq'].keys()
