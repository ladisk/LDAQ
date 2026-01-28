"""
Tests for LDAQ Core class.

This module tests the Core orchestrator that coordinates multiple acquisition sources.
Note: Core.run() requires root on Linux due to keyboard library, so tests use
acquisition.run_acquisition() directly.

See openspec/changes/test-core/specs/ for detailed requirements.
"""

import pytest
import numpy as np

import LDAQ


# =============================================================================
# Initialization Tests
# =============================================================================

class TestCoreInitialization:
    """Tests for Core initialization."""

    def test_single_acquisition_source(self, simple_simulated_acquisition):
        """Test Core initializes with single acquisition source.

        Spec: Core initialization - Single acquisition source
        """
        core = LDAQ.Core(acquisitions=simple_simulated_acquisition)

        # Contract: acquisitions is list with one element
        assert isinstance(core.acquisitions, list)
        assert len(core.acquisitions) == 1
        assert core.acquisition_names == ['test_acq']

    def test_multiple_acquisition_sources(self):
        """Test Core initializes with multiple acquisition sources.

        Spec: Core initialization - Multiple acquisition sources
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)

        core = LDAQ.Core(acquisitions=[acq1, acq2])

        # Contract: acquisitions has 2 elements
        assert len(core.acquisitions) == 2
        assert core.acquisition_names == ['source1', 'source2']

    def test_duplicate_names_raises_exception(self):
        """Test Core raises Exception for duplicate acquisition names.

        Spec: Core initialization - Duplicate names raises error
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='same_name')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='same_name')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)

        with pytest.raises(Exception):
            LDAQ.Core(acquisitions=[acq1, acq2])


# =============================================================================
# Trigger Cascade Tests
# =============================================================================

class TestTriggerCascade:
    """Tests for trigger cascade across sources."""

    def test_duration_cascades_to_all_sources(self):
        """Test duration cascades to all sources.

        Spec: Trigger cascade - Duration cascades to all sources
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)

        core = LDAQ.Core(acquisitions=[acq1, acq2])
        core.set_trigger(source=0, channel=0, level=1.0, duration=2.0)

        # Contract: both sources have 2.0 seconds = 2000 samples at 1kHz
        assert acq1.trigger_settings['duration_samples'] == 2000
        assert acq2.trigger_settings['duration_samples'] == 2000

    def test_presamples_converted_across_sample_rates(self):
        """Test presamples converted across different sample rates.

        Spec: Trigger cascade - Presamples converted across sample rates
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)  # 1kHz
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=500)   # 500Hz

        core = LDAQ.Core(acquisitions=[acq1, acq2])
        core.set_trigger(source=0, channel=0, level=1.0, duration=1.0, presamples=100)

        # Contract: 100 presamples at 1kHz = 0.1s = 50 samples at 500Hz
        assert acq1.trigger_settings['presamples'] == 100
        assert acq2.trigger_settings['presamples'] == 50

    def test_trigger_source_index_stored(self):
        """Test trigger source index is stored.

        Spec: Trigger cascade - Trigger source index stored
        """
        acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
        acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

        data = np.zeros((100, 1))
        acq1.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
        acq2.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)

        core = LDAQ.Core(acquisitions=[acq1, acq2])
        core.set_trigger(source=1, channel=0, level=1.0, duration=1.0)

        # Contract: trigger_source_index should be 1
        assert core.trigger_source_index == 1


# =============================================================================
# Measurement Dict Tests
# =============================================================================

class TestMeasurementDict:
    """Tests for get_measurement_dict()."""

    def test_returns_dict_keyed_by_acquisition_name(self, simple_simulated_acquisition):
        """Test returns dict keyed by acquisition name.

        Spec: Measurement dict structure - Returns dict keyed by acquisition name
        """
        core = LDAQ.Core(acquisitions=simple_simulated_acquisition)
        acq = core.acquisitions[0]
        acq.run_acquisition(0.1)

        result = core.get_measurement_dict()

        # Contract: dict with acquisition name as key
        assert isinstance(result, dict)
        assert 'test_acq' in result

    def test_each_value_satisfies_measurement_contract(self, simple_simulated_acquisition, validate_measurement_dict):
        """Test each value satisfies measurement contract.

        Spec: Measurement dict structure - Each value satisfies measurement contract
        """
        core = LDAQ.Core(acquisitions=simple_simulated_acquisition)
        acq = core.acquisitions[0]
        acq.run_acquisition(0.1)

        result = core.get_measurement_dict()

        # Contract: each source's dict has required fields
        source_data = result['test_acq']
        assert 'time' in source_data
        assert 'data' in source_data
        assert 'channel_names' in source_data
        assert 'sample_rate' in source_data

        # Use validator for full contract check
        validate_measurement_dict(source_data, source_level=True)


# =============================================================================
# Save Measurement Tests
# =============================================================================

class TestSaveMeasurement:
    """Tests for save_measurement()."""

    def test_creates_pickle_file(self, simple_simulated_acquisition, temp_measurement_dir):
        """Test save_measurement creates pickle file.

        Spec: Save measurement - Creates pickle file
        """
        core = LDAQ.Core(acquisitions=simple_simulated_acquisition)
        acq = core.acquisitions[0]
        acq.run_acquisition(0.1)

        core.save_measurement(name='test_save', root=str(temp_measurement_dir), timestamp=False)

        # Contract: file should exist
        expected_file = temp_measurement_dir / 'test_save.pkl'
        assert expected_file.exists()

    def test_file_is_loadable(self, simple_simulated_acquisition, temp_measurement_dir):
        """Test saved file is loadable with correct structure.

        Spec: Save measurement - File is loadable
        """
        core = LDAQ.Core(acquisitions=simple_simulated_acquisition)
        acq = core.acquisitions[0]
        acq.run_acquisition(0.1)

        core.save_measurement(name='test_load', root=str(temp_measurement_dir), timestamp=False)

        # Load and verify
        loaded = LDAQ.load_measurement('test_load.pkl', directory=str(temp_measurement_dir))

        # Contract: loaded dict has acquisition name as key
        assert 'test_acq' in loaded
        assert 'time' in loaded['test_acq']
        assert 'data' in loaded['test_acq']
