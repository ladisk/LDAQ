"""
Tests for BaseAcquisition class.

This module tests the BaseAcquisition functionality through SimulatedAcquisition,
which inherits from BaseAcquisition and exercises all its methods.

Tests cover:
- Channel name management (data, video, all)
- Trigger configuration (set_trigger, update_trigger_parameters)
- Data retrieval (get_data, get_measurement_dict, _reshape_data)
- Virtual channel computation
- Continuous mode behavior
- Acquisition lifecycle methods

See openspec/changes/test-acquisition-base/specs/ for detailed requirements.
"""

import pytest
import numpy as np

import LDAQ
from LDAQ.acquisition_base import CustomPyTrigger


# =============================================================================
# Channel Management Tests
# =============================================================================

class TestChannelManagement:
    """Tests for channel name management in BaseAcquisition."""

    def test_data_channels_populated_correctly(self, multi_channel_acquisition):
        """Test that data channels are correctly populated.

        Spec: Channel name management - Data channels populated correctly
        """
        acq = multi_channel_acquisition

        # Contract: 3 channels configured in fixture: ch0, ch1, ch2
        assert acq.channel_names == ['ch0', 'ch1', 'ch2']
        assert acq.channel_names_all == ['ch0', 'ch1', 'ch2']
        assert acq.channel_names_video == []

    def test_video_channels_populated_correctly(self, video_acquisition):
        """Test that video channels are correctly populated.

        Spec: Channel name management - Video channels populated correctly
        """
        acq = video_acquisition

        # Contract: 1 video channel configured in fixture: camera
        assert acq.channel_names_video == ['camera']
        assert acq.channel_names_all == ['camera']
        assert acq.channel_names == []

    def test_get_channel_index_data_type(self, multi_channel_acquisition):
        """Test get_channel_index returns correct index for data type.

        Spec: Channel name management - get_channel_index returns correct index for data
        """
        acq = multi_channel_acquisition

        # Contract: ch1 is at index 1 in ['ch0', 'ch1', 'ch2']
        assert acq.get_channel_index('ch1', channel_type='data') == 1
        assert acq.get_channel_index('ch0', channel_type='data') == 0
        assert acq.get_channel_index('ch2', channel_type='data') == 2

    def test_get_channel_index_all_type(self, multi_channel_acquisition):
        """Test get_channel_index returns correct index for 'all' type.

        Spec: Channel name management - get_channel_index returns correct index for all
        """
        acq = multi_channel_acquisition

        # Contract: ch1 is at index 1 in channel_names_all
        assert acq.get_channel_index('ch1', channel_type='all') == 1

    def test_get_channel_index_raises_for_invalid(self, multi_channel_acquisition):
        """Test get_channel_index raises ValueError for invalid channel.

        Spec: Channel name management - get_channel_index raises for invalid channel
        """
        acq = multi_channel_acquisition

        with pytest.raises(ValueError):
            acq.get_channel_index('nonexistent', channel_type='data')


# =============================================================================
# Trigger Configuration Tests
# =============================================================================

class TestTriggerConfiguration:
    """Tests for trigger configuration via set_trigger()."""

    def test_set_trigger_with_seconds_duration(self, simple_simulated_acquisition):
        """Test set_trigger with seconds duration unit.

        Spec: Trigger configuration - set_trigger with seconds duration
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=1.0, channel=0, duration=2.0, duration_unit='seconds')

        # Contract: 1kHz sample rate, 2.0 seconds = 2000 samples
        assert acq.trigger_settings['duration_samples'] == 2000
        assert acq.trigger_settings['duration_seconds'] == 2.0
        assert acq.trigger_settings['level'] == 1.0

    def test_set_trigger_with_samples_duration(self, simple_simulated_acquisition):
        """Test set_trigger with samples duration unit.

        Spec: Trigger configuration - set_trigger with samples duration
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=0.5, channel='ch0', duration=500, duration_unit='samples')

        # Contract: 500 samples at 1kHz = 0.5 seconds
        assert acq.trigger_settings['duration_samples'] == 500
        assert acq.trigger_settings['duration_seconds'] == 0.5

    def test_set_trigger_with_presamples(self, simple_simulated_acquisition):
        """Test set_trigger with presamples parameter.

        Spec: Trigger configuration - set_trigger with presamples
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=1.0, channel=0, duration=1.0, presamples=100)

        # Contract: presamples should be 100 in both settings and Trigger
        assert acq.trigger_settings['presamples'] == 100
        assert acq.Trigger.presamples == 100

    def test_set_trigger_creates_trigger_instance(self, simple_simulated_acquisition):
        """Test set_trigger creates Trigger instance.

        Spec: Trigger configuration - set_trigger creates Trigger instance
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=1.0, channel=0, duration=1.0)

        # Contract: Trigger attribute should exist and be CustomPyTrigger
        assert hasattr(acq, 'Trigger')
        assert isinstance(acq.Trigger, CustomPyTrigger)


# =============================================================================
# Trigger Parameter Update Tests
# =============================================================================

class TestTriggerParameterUpdates:
    """Tests for update_trigger_parameters()."""

    def test_update_trigger_parameters_changes_duration(self, simple_simulated_acquisition):
        """Test update_trigger_parameters changes duration.

        Spec: Trigger parameter updates - update_trigger_parameters changes duration
        """
        acq = simple_simulated_acquisition

        # Initial trigger
        acq.set_trigger(level=1.0, channel=0, duration=1.0, duration_unit='seconds')

        # Update duration
        acq.update_trigger_parameters(duration=3.0)

        # Contract: 3.0 seconds at 1kHz = 3000 samples
        assert acq.trigger_settings['duration'] == 3.0
        assert acq.trigger_settings['duration_samples'] == 3000

    def test_update_trigger_parameters_changes_level(self, simple_simulated_acquisition):
        """Test update_trigger_parameters changes level.

        Spec: Trigger parameter updates - update_trigger_parameters changes level
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=1.0, channel=0, duration=1.0)
        acq.update_trigger_parameters(level=2.5)

        # Contract: level should be updated to 2.5
        assert acq.trigger_settings['level'] == 2.5

    def test_update_trigger_parameters_preserves_unchanged(self, simple_simulated_acquisition):
        """Test update_trigger_parameters preserves unchanged settings.

        Spec: Trigger parameter updates - update_trigger_parameters preserves unchanged settings
        """
        acq = simple_simulated_acquisition

        acq.set_trigger(level=1.0, channel=0, duration=1.0, presamples=50)
        acq.update_trigger_parameters(level=2.0)

        # Contract: presamples should remain 50
        assert acq.trigger_settings['presamples'] == 50


# =============================================================================
# Data Retrieval Tests
# =============================================================================

class TestDataRetrieval:
    """Tests for get_data() method."""

    def test_get_data_returns_tuple(self, simple_simulated_acquisition):
        """Test get_data returns tuple of (time, data).

        Spec: Data retrieval - get_data returns tuple of time and data
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)  # 100ms at 1kHz

        result = acq.get_data()

        # Contract: returns tuple of (time, data)
        assert isinstance(result, tuple)
        assert len(result) == 2

        time, data = result
        assert isinstance(time, np.ndarray)
        assert time.ndim == 1
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2

    def test_get_data_with_n_points(self, simple_simulated_acquisition):
        """Test get_data with N_points parameter limits rows.

        Spec: Data retrieval - get_data with N_points parameter
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)  # 100ms

        time, data = acq.get_data(N_points=50)

        # Contract: should return exactly 50 rows
        assert data.shape[0] == 50
        assert time.shape[0] == 50

    def test_get_data_flattened(self, simple_simulated_acquisition):
        """Test get_data with data_to_return='flattened'.

        Spec: Data retrieval - get_data with data_to_return='flattened'
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        time, data = acq.get_data(data_to_return='flattened')

        # Contract: flattened returns all ring buffer channels
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2


# =============================================================================
# Measurement Dictionary Tests
# =============================================================================

class TestMeasurementDictionary:
    """Tests for get_measurement_dict() method."""

    def test_required_fields_data_acquisition(self, simple_simulated_acquisition, validate_measurement_dict):
        """Test required fields present for data acquisition.

        Spec: Measurement dictionary - Required fields present for data acquisition
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        result = acq.get_measurement_dict()

        # Contract: required fields present
        assert 'time' in result
        assert 'data' in result
        assert 'channel_names' in result
        assert 'sample_rate' in result

        # Contract: correct types and shapes
        assert isinstance(result['time'], np.ndarray)
        assert result['time'].ndim == 1
        assert isinstance(result['data'], np.ndarray)
        assert result['data'].ndim == 2
        assert isinstance(result['channel_names'], list)
        assert len(result['channel_names']) == result['data'].shape[1]

        # Use validator fixture
        validate_measurement_dict(result, source_level=True)

    def test_video_fields_present(self, video_acquisition, validate_measurement_dict):
        """Test video fields present for video acquisition.

        Spec: Measurement dictionary - Video fields present for video acquisition
        """
        acq = video_acquisition
        acq.run_acquisition(0.1)  # 100ms at 30fps = ~3 frames

        result = acq.get_measurement_dict()

        # Contract: video fields present
        assert 'video' in result
        assert 'channel_names_video' in result

        # Contract: correct types
        assert isinstance(result['video'], list)
        assert len(result['video']) > 0
        assert result['video'][0].ndim == 3

        # Contract: first dimension matches time length
        assert result['video'][0].shape[0] == result['time'].shape[0]

        # Use validator fixture
        validate_measurement_dict(result, source_level=True)

    def test_sample_rate_populated(self, simple_simulated_acquisition):
        """Test sample_rate field populated correctly.

        Spec: Measurement dictionary - Sample rate field populated
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        result = acq.get_measurement_dict()

        # Contract: sample_rate should be 1000 (from fixture)
        assert result['sample_rate'] == 1000


# =============================================================================
# Data Reshaping Tests
# =============================================================================

class TestDataReshaping:
    """Tests for _reshape_data() method."""

    def test_reshape_data_extracts_data_channels(self, multi_channel_acquisition):
        """Test _reshape_data extracts data channels correctly.

        Spec: Data reshaping - _reshape_data extracts data channels
        """
        acq = multi_channel_acquisition
        acq.run_acquisition(0.1)

        # Get flattened data from ring buffer
        flattened = acq.Trigger.get_data()

        result = acq._reshape_data(flattened, data_to_return='data')

        # Contract: 2D array with shape (n_samples, n_data_channels)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 3  # 3 data channels in fixture

    def test_reshape_data_extracts_video_channels(self, video_acquisition):
        """Test _reshape_data extracts video channels as list of 3D arrays.

        Spec: Data reshaping - _reshape_data extracts video channels
        """
        acq = video_acquisition
        acq.run_acquisition(0.1)

        flattened = acq.Trigger.get_data()

        result = acq._reshape_data(flattened, data_to_return='video')

        # Contract: list of 3D arrays with shape (n_samples, height, width)
        assert isinstance(result, list)
        assert len(result) == 1  # 1 video channel
        assert result[0].ndim == 3
        assert result[0].shape[1:] == (32, 32)  # 32x32 from fixture

    def test_reshape_data_raises_for_no_video(self, simple_simulated_acquisition):
        """Test _reshape_data raises ValueError for video on data-only acquisition.

        Spec: Data reshaping - _reshape_data raises for no video channels
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        flattened = acq.Trigger.get_data()

        with pytest.raises(ValueError):
            acq._reshape_data(flattened, data_to_return='video')


# =============================================================================
# Virtual Channel Tests
# =============================================================================

class TestVirtualChannels:
    """Tests for virtual channel computation."""

    def test_virtual_channel_added_to_lists(self, multi_channel_acquisition):
        """Test virtual channel added to channel_names and channel_names_all.

        Spec: Virtual channel computation - Virtual channel added to channel lists
        """
        acq = multi_channel_acquisition

        def sum_channels(ch0, ch1):
            return (ch0 + ch1).reshape(-1, 1)

        acq.add_virtual_channel('sum', ['ch0', 'ch1'], sum_channels)

        # Contract: 'sum' should appear in channel lists
        assert 'sum' in acq.channel_names
        assert 'sum' in acq.channel_names_all

    def test_virtual_channel_computed_correctly(self, acquisition_with_virtual_channel, validate_measurement_dict):
        """Test virtual channel computed correctly (ratio example).

        Spec: Virtual channel computation - Virtual channel computed correctly
        """
        acq = acquisition_with_virtual_channel
        acq.run_acquisition(0.1)

        result = acq.get_measurement_dict()

        # Contract: ratio column should be ch0/ch1
        ch0_idx = result['channel_names'].index('ch0')
        ch1_idx = result['channel_names'].index('ch1')
        ratio_idx = result['channel_names'].index('ratio')

        ch0_data = result['data'][:, ch0_idx]
        ch1_data = result['data'][:, ch1_idx]
        ratio_data = result['data'][:, ratio_idx]

        expected_ratio = ch0_data / ch1_data
        np.testing.assert_array_almost_equal(ratio_data, expected_ratio)

        validate_measurement_dict(result, source_level=True)

    def test_virtual_channel_with_self_reference(self, multi_channel_acquisition):
        """Test virtual channel with self reference receives acquisition instance.

        Spec: Virtual channel computation - Virtual channel with self reference
        """
        acq = multi_channel_acquisition

        received_self = []

        def func_with_self(self, ch0):
            received_self.append(self)
            return ch0.reshape(-1, 1)

        acq.add_virtual_channel('with_self', ['ch0'], func_with_self)
        acq.run_acquisition(0.05)

        # Contract: function should receive acquisition instance
        assert len(received_self) > 0
        assert received_self[0] is acq

    def test_virtual_channel_non_array_raises(self, multi_channel_acquisition):
        """Test virtual channel function returning non-array raises ValueError.

        Spec: Virtual channel computation - Virtual channel function must return numpy array
        """
        acq = multi_channel_acquisition

        def returns_scalar(ch0):
            return 42  # scalar, not array

        with pytest.raises(ValueError):
            acq.add_virtual_channel('bad', ['ch0'], returns_scalar)


# =============================================================================
# Continuous Mode Tests
# =============================================================================

class TestContinuousMode:
    """Tests for continuous mode configuration."""

    def test_set_continuous_mode_enables_flag(self, simple_simulated_acquisition):
        """Test set_continuous_mode enables continuous flag.

        Spec: Continuous mode - set_continuous_mode enables continuous flag
        """
        acq = simple_simulated_acquisition

        acq.set_continuous_mode(True)

        # Contract: continuous_mode should be True
        assert acq.continuous_mode is True

    def test_set_continuous_mode_with_duration(self, simple_simulated_acquisition):
        """Test set_continuous_mode with duration sets N_samples_to_acquire.

        Spec: Continuous mode - set_continuous_mode with duration sets N_samples_to_acquire
        """
        acq = simple_simulated_acquisition

        acq.set_continuous_mode(True, measurement_duration=5.0)

        # Contract: 5.0 seconds at 1kHz = 5000 samples
        assert acq.N_samples_to_acquire == 5000

    def test_set_continuous_mode_without_duration(self, simple_simulated_acquisition):
        """Test set_continuous_mode without duration allows indefinite.

        Spec: Continuous mode - set_continuous_mode without duration allows indefinite
        """
        acq = simple_simulated_acquisition

        acq.set_continuous_mode(True, measurement_duration=None)

        # Contract: N_samples_to_acquire should be None
        assert acq.N_samples_to_acquire is None

    def test_continuous_mode_propagates_to_trigger(self, simple_simulated_acquisition):
        """Test continuous mode propagates to Trigger instance.

        Spec: Continuous mode - Continuous mode propagates to Trigger
        """
        acq = simple_simulated_acquisition

        acq.set_continuous_mode(True)
        acq.set_trigger(level=1.0, channel=0, duration=1.0)

        # Contract: Trigger.continuous_mode should be True
        assert acq.Trigger.continuous_mode is True


# =============================================================================
# Lifecycle Method Tests
# =============================================================================

class TestLifecycleMethods:
    """Tests for acquisition lifecycle methods."""

    def test_stop_sets_is_running_false(self, simple_simulated_acquisition):
        """Test stop() sets is_running to False.

        Spec: Acquisition lifecycle - stop sets is_running to False
        """
        acq = simple_simulated_acquisition
        acq.is_running = True

        acq.stop()

        # Contract: is_running should be False
        assert acq.is_running is False

    def test_is_triggered_returns_state(self, simple_simulated_acquisition):
        """Test is_triggered() returns trigger state.

        Spec: Acquisition lifecycle - is_triggered returns trigger state
        """
        acq = simple_simulated_acquisition
        acq.set_trigger(level=0, channel=0, duration=0.1)  # level=0 triggers immediately
        acq.run_acquisition(0.05)

        # Contract: should return True after being triggered
        assert acq.is_triggered() is True

    def test_activate_trigger_sets_global(self, simple_simulated_acquisition):
        """Test activate_trigger sets triggered_global.

        Spec: Acquisition lifecycle - activate_trigger triggers all sources
        """
        acq = simple_simulated_acquisition

        # Reset first
        CustomPyTrigger.triggered_global = False

        acq.activate_trigger(all_sources=True)

        # Contract: triggered_global should be True
        assert CustomPyTrigger.triggered_global is True

        # Cleanup
        CustomPyTrigger.triggered_global = False

    def test_reset_trigger_clears_state(self, simple_simulated_acquisition):
        """Test reset_trigger clears trigger state.

        Spec: Acquisition lifecycle - reset_trigger clears trigger state
        """
        acq = simple_simulated_acquisition
        acq.set_trigger(level=0, channel=0, duration=0.1)
        acq.run_acquisition(0.05)

        # Trigger should be set
        assert acq.Trigger.triggered is True

        acq.reset_trigger()

        # Contract: both should be False after reset
        assert acq.Trigger.triggered is False
        assert CustomPyTrigger.triggered_global is False
