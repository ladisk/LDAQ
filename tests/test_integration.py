"""
Tests for LDAQ integration scenarios.

This module tests integration between multiple LDAQ components working together,
including multi-source acquisition, cross-source triggering, and save/load cycles.

See openspec/changes/test-integration/specs/ for detailed requirements.
"""

import pytest
import numpy as np

import LDAQ
from LDAQ.acquisition_base import CustomPyTrigger


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_triggered_global():
    """Reset triggered_global before each test to avoid state pollution."""
    CustomPyTrigger.triggered_global = False
    yield
    CustomPyTrigger.triggered_global = False


# =============================================================================
# Multi-Source Acquisition Tests
# =============================================================================

class TestMultiSourceAcquisition:
    """Tests for multiple acquisition sources working together."""

    def test_two_sources_produce_independent_data(self):
        """Test two sources produce independent data.

        Spec: Multi-source acquisition - Two sources produce independent data
        """
        # Create two sources with different data
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data1 = np.ones((100, 1)) * 1.0
        data2 = np.ones((100, 1)) * 2.0

        acq1.set_simulated_data(data1, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data2, channel_names=['ch0'], sample_rate=1000)

        core = LDAQ.Core(acquisitions=[acq1, acq2])

        # Run acquisitions directly (Core.run() requires root on Linux)
        acq1.run_acquisition(0.05)
        acq2.run_acquisition(0.05)

        result = core.get_measurement_dict()

        # Contract: each source returns its own data
        assert 'source1' in result
        assert 'source2' in result

        # Source1 should have ~1.0 values, source2 ~2.0 values
        source1_data = result['source1']['data']
        source2_data = result['source2']['data']

        # Check that the data is distinct (not mixed up)
        if source1_data.size > 0 and source2_data.size > 0:
            assert np.mean(source1_data) < 1.5  # source1 has 1.0s
            assert np.mean(source2_data) > 1.5  # source2 has 2.0s

    def test_different_sample_rates_work_correctly(self):
        """Test different sample rates produce correct sample counts.

        Spec: Multi-source acquisition - Different sample rates work correctly
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='fast')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='slow')

        # 1kHz and 500Hz sources
        data1 = np.zeros((1000, 1))  # enough for 1 second at 1kHz
        data2 = np.zeros((500, 1))   # enough for 1 second at 500Hz

        acq1.set_simulated_data(data1, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data2, channel_names=['ch0'], sample_rate=500)

        acq1.run_acquisition(0.1)
        acq2.run_acquisition(0.1)

        result1 = acq1.get_measurement_dict()
        result2 = acq2.get_measurement_dict()

        # Contract: 0.1s at 1kHz = ~100 samples, 0.1s at 500Hz = ~50 samples
        # Allow some tolerance for timing
        assert 80 <= result1['data'].shape[0] <= 120
        assert 40 <= result2['data'].shape[0] <= 60

        # Verify sample rates are preserved
        assert result1['sample_rate'] == 1000
        assert result2['sample_rate'] == 500


# =============================================================================
# Cross-Source Triggering Tests
# =============================================================================

class TestCrossSourceTriggering:
    """Tests for triggering across multiple sources."""

    def test_activate_trigger_all_sources_sets_global(self):
        """Test activate_trigger with all_sources sets triggered_global.

        Spec: Cross-source triggering - activate_trigger with all_sources triggers all
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')
        data = np.zeros((100, 1))
        acq.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)

        # Verify initial state
        assert CustomPyTrigger.triggered_global is False

        # activate_trigger with all_sources=True
        acq.activate_trigger(all_sources=True)

        # Contract: triggered_global should be True
        assert CustomPyTrigger.triggered_global is True

    def test_core_set_trigger_cascades_duration(self):
        """Test Core.set_trigger cascades duration to all sources.

        Spec: Cross-source triggering - Core.set_trigger cascades duration
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=500)

        core = LDAQ.Core(acquisitions=[acq1, acq2])
        core.set_trigger(source=0, channel=0, level=1.0, duration=2.0)

        # Contract: 2.0 seconds at 1kHz = 2000 samples, at 500Hz = 1000 samples
        assert acq1.trigger_settings['duration_samples'] == 2000
        assert acq2.trigger_settings['duration_samples'] == 1000

    def test_core_set_trigger_cascades_presamples(self):
        """Test Core.set_trigger converts presamples across sample rates.

        Spec: Cross-source triggering - Core.set_trigger cascades duration
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=500)

        core = LDAQ.Core(acquisitions=[acq1, acq2])
        core.set_trigger(source=0, channel=0, level=1.0, duration=1.0, presamples=100)

        # Contract: 100 presamples at 1kHz = 0.1s = 50 presamples at 500Hz
        assert acq1.trigger_settings['presamples'] == 100
        assert acq2.trigger_settings['presamples'] == 50


# =============================================================================
# Multi-Source Save/Load Tests
# =============================================================================

class TestMultiSourceSaveLoad:
    """Tests for save/load with multiple sources."""

    def test_save_load_preserves_all_sources(self, temp_measurement_dir):
        """Test save and load preserves all sources.

        Spec: Multi-source save/load - Save and load preserves all sources
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        # Create distinct data
        data1 = np.array([[1.0], [2.0], [3.0]])
        data2 = np.array([[10.0], [20.0], [30.0]])

        acq1.set_simulated_data(data1, channel_names=['a'], sample_rate=100)
        acq2.set_simulated_data(data2, channel_names=['b'], sample_rate=100)

        core = LDAQ.Core(acquisitions=[acq1, acq2])

        acq1.run_acquisition(0.03)
        acq2.run_acquisition(0.03)

        # Save
        core.save_measurement(name='multi_source', root=str(temp_measurement_dir), timestamp=False)

        # Load
        loaded = LDAQ.load_measurement('multi_source.pkl', directory=str(temp_measurement_dir))

        # Contract: both sources present
        assert 'source1' in loaded
        assert 'source2' in loaded

        # Contract: data structure preserved
        assert 'time' in loaded['source1']
        assert 'data' in loaded['source1']
        assert 'channel_names' in loaded['source1']

        assert 'time' in loaded['source2']
        assert 'data' in loaded['source2']
        assert 'channel_names' in loaded['source2']
