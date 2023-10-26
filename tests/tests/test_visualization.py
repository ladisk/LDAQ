import os
import sys
sys.path.insert(0, os.path.realpath('../../'))
import time

import LDAQ


# def test_Visualization_basic():
#     vis = LDAQ.Visualization()
#     vis.add_lines((0, 0), source='NI_acq', channels=['ch1', 'ch2'])
#     vis.add_lines((0, 1), source='NI_acq', channels=['ch1', 1])

#     vis.add_image('NI_acq', 'image1')
#     vis.add_image('NI_acq', 1)

#     assert 'NI_acq' in vis.plots.keys()
#     assert vis.plots['NI_acq'][0]['pos'] == (0, 0)
#     assert vis.plots['NI_acq'][1]['pos'] == (0, 0)
#     assert vis.plots['NI_acq'][2]['pos'] == (0, 1)
#     assert vis.plots['NI_acq'][3]['pos'] == (0, 1)

#     assert vis.plots['NI_acq'][0]['channels'] == 'ch1'
#     assert vis.plots['NI_acq'][1]['channels'] == 'ch2'
#     assert vis.plots['NI_acq'][2]['channels'] == 'ch1'
#     assert vis.plots['NI_acq'][3]['channels'] == 1


# def test_Visualization_basic_2():
#     vis = LDAQ.Visualization()
#     vis.add_lines((0, 0), source='NI_acq', channels=['ch1', 'ch2'])
#     vis.add_lines((0, 1), source='NI_acq', channels=['ch1', 1])

#     vis.config_subplot((0, 0), xlim=(0, 10), ylim=(0, 10), title='Test title', axis_style='semilogy')

#     assert (0, 0) in vis.subplot_options.keys()
#     assert vis.subplot_options[(0, 0)]['xlim'] == (0, 10)
#     assert vis.subplot_options[(0, 0)]['ylim'] == (0, 10)
#     assert vis.subplot_options[(0, 0)]['title'] == 'Test title'
#     assert vis.subplot_options[(0, 0)]['axis_style'] == 'semilogy'
#     assert vis.subplot_options[(0, 0)]['colspan'] == 1
#     assert vis.subplot_options[(0, 0)]['rowspan'] == 1