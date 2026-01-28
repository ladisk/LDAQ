"""
Tests for SimulatedAcquisition class.

This module tests the SimulatedAcquisition functionality including:
- Data simulation with numpy arrays (looped playback)
- Data simulation with callable functions
- Video simulation with 3D arrays
- Channel name auto-generation
- Sample rate configuration
- Input validation
- Threading mode operation

See openspec/changes/test-simulator/specs/ for detailed requirements.
"""

import pytest
import numpy as np

import LDAQ


# =============================================================================
# Data Simulation with Array Tests
# =============================================================================

class TestDataSimulationWithArray:
    """Tests for set_simulated_data() with numpy array input."""

    def test_array_data_looped_correctly(self):
        """Test that array data is looped correctly during acquisition.

        Spec: Data simulation with numpy array - Array data is looped correctly
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        # Create small array that will loop
        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 samples, 2 channels
        acq.set_simulated_data(input_data, channel_names=['a', 'b'], sample_rate=100)

        # Run long enough to get multiple loops
        acq.run_acquisition(0.1)  # 100ms at 100Hz = ~10 samples

        result = acq.get_measurement_dict()

        # Contract: data should contain values from input array (looped)
        # Check that all values in result are from the input array
        unique_col0 = set(np.round(result['data'][:, 0], 5))
        unique_col1 = set(np.round(result['data'][:, 1], 5))

        assert unique_col0.issubset({1.0, 3.0, 5.0})
        assert unique_col1.issubset({2.0, 4.0, 6.0})

    def test_channel_names_from_parameter(self, simple_simulated_acquisition):
        """Test that channel names are set from parameter.

        Spec: Data simulation with numpy array - Channel names set from parameter
        """
        acq = simple_simulated_acquisition

        # Contract: fixture sets channel_names=['ch0']
        assert acq.channel_names == ['ch0']

    def test_channel_names_auto_generated(self):
        """Test that channel names are auto-generated when not provided.

        Spec: Data simulation with numpy array - Channel names auto-generated when not provided
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        data = np.zeros((100, 3))  # 3 channels
        acq.set_simulated_data(data, channel_names=None, sample_rate=1000)

        # Contract: auto-generated names are channel_0, channel_1, channel_2
        assert acq.channel_names == ['channel_0', 'channel_1', 'channel_2']

    def test_sample_rate_defaults_to_1000(self):
        """Test that sample rate defaults to 1000 Hz.

        Spec: Data simulation with numpy array - Sample rate defaults to 1000 Hz
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        data = np.zeros((100, 1))
        acq.set_simulated_data(data, channel_names=['ch0'], sample_rate=None)

        # Contract: default sample rate is 1000 Hz
        assert acq.sample_rate == 1000

    def test_custom_sample_rate(self):
        """Test that custom sample rate is used.

        Spec: Data simulation with numpy array - Custom sample rate is used
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        data = np.zeros((100, 1))
        acq.set_simulated_data(data, channel_names=['ch0'], sample_rate=500)

        # Contract: custom sample rate is used
        assert acq.sample_rate == 500


# =============================================================================
# Data Simulation with Function Tests
# =============================================================================

class TestDataSimulationWithFunction:
    """Tests for set_simulated_data() with callable function input."""

    def test_function_receives_time_array_and_args(self):
        """Test that function receives time array and args.

        Spec: Data simulation with function - Function receives time array
        """
        received_args = []

        def test_func(t, arg1, arg2):
            received_args.append((t, arg1, arg2))
            return np.column_stack([np.sin(t), np.cos(t)])

        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')
        acq.set_simulated_data(test_func, channel_names=['sin', 'cos'], sample_rate=100, args=(42, 'hello'))

        # Contract: function is called during set_simulated_data to validate output
        assert len(received_args) > 0
        t, arg1, arg2 = received_args[0]

        # Time array should be numpy array
        assert isinstance(t, np.ndarray)
        # Args should be passed correctly
        assert arg1 == 42
        assert arg2 == 'hello'

    def test_function_output_shapes_data(self):
        """Test that function output shapes data correctly.

        Spec: Data simulation with function - Function output shapes data correctly
        """
        def generate_data(t):
            return np.column_stack([t, t**2, t**3])  # 3 channels

        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')
        acq.set_simulated_data(generate_data, channel_names=['t', 't2', 't3'], sample_rate=100)

        acq.run_acquisition(0.1)
        result = acq.get_measurement_dict()

        # Contract: data has 3 channels as returned by function
        assert result['data'].shape[1] == 3
        assert result['channel_names'] == ['t', 't2', 't3']


# =============================================================================
# Video Simulation Tests
# =============================================================================

class TestVideoSimulation:
    """Tests for set_simulated_video() with 3D array input."""

    def test_video_data_configured_correctly(self, video_acquisition):
        """Test that video data is configured correctly.

        Spec: Video simulation with numpy array - Video data configured correctly
        """
        acq = video_acquisition

        # Contract: fixture configures 32x32 video with channel 'camera'
        assert 'camera' in acq.channel_names_video
        assert acq._channel_shapes_video_init == [(32, 32)]

    def test_video_channel_name_defaults(self):
        """Test that video channel name defaults to 'video_channel'.

        Spec: Video simulation with numpy array - Video channel name defaults to 'video_channel'
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        video = np.zeros((10, 8, 8), dtype=np.uint8)  # 10 frames, 8x8
        acq.set_simulated_video(video, channel_name_video=None, sample_rate=30)

        # Contract: default channel name is 'video_channel'
        assert acq.channel_names_video == ['video_channel']

    def test_custom_video_channel_name(self):
        """Test that custom video channel name is used.

        Spec: Video simulation with numpy array - Custom video channel name is used
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        video = np.zeros((10, 8, 8), dtype=np.uint8)
        acq.set_simulated_video(video, channel_name_video='thermal_cam', sample_rate=30)

        # Contract: custom channel name is used
        assert acq.channel_names_video == ['thermal_cam']

    def test_video_sample_rate_defaults_to_30(self):
        """Test that video sample rate defaults to 30 Hz.

        Spec: Video simulation with numpy array - Video sample rate defaults to 30 Hz
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        video = np.zeros((10, 8, 8), dtype=np.uint8)
        acq.set_simulated_video(video, channel_name_video='cam', sample_rate=None)

        # Contract: default sample rate is 30 Hz
        assert acq.sample_rate == 30


# =============================================================================
# Data/Video Exclusivity Tests
# =============================================================================

class TestDataVideoExclusivity:
    """Tests documenting that data and video channels cannot be combined."""

    def test_set_simulated_video_clears_data_channels(self):
        """Test that set_simulated_video clears data channels.

        Spec: Data and video channel exclusivity - set_simulated_video clears data channels
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        # First set data
        data = np.zeros((100, 2))
        acq.set_simulated_data(data, channel_names=['ch0', 'ch1'], sample_rate=1000)
        assert acq.channel_names == ['ch0', 'ch1']

        # Then set video - this should clear data channels
        video = np.zeros((10, 8, 8), dtype=np.uint8)
        acq.set_simulated_video(video, channel_name_video='cam', sample_rate=30)

        # Contract: data channels should be empty
        assert acq.channel_names == []
        assert acq.channel_names_video == ['cam']

    def test_set_simulated_data_clears_video_channels(self):
        """Test that set_simulated_data clears video channels.

        Spec: Data and video channel exclusivity - set_simulated_data clears video channels
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        # First set video
        video = np.zeros((10, 8, 8), dtype=np.uint8)
        acq.set_simulated_video(video, channel_name_video='cam', sample_rate=30)
        assert acq.channel_names_video == ['cam']

        # Then set data - this should clear video channels
        data = np.zeros((100, 2))
        acq.set_simulated_data(data, channel_names=['ch0', 'ch1'], sample_rate=1000)

        # Contract: video channels should be empty
        assert acq.channel_names_video == []
        assert acq.channel_names == ['ch0', 'ch1']


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation in SimulatedAcquisition."""

    def test_non_2d_array_rejected_for_data(self):
        """Test that non-2D array is rejected for data.

        Spec: Input validation - Non-2D array rejected for data
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        data_1d = np.zeros(100)  # 1D array

        with pytest.raises(ValueError):
            acq.set_simulated_data(data_1d, channel_names=['ch0'], sample_rate=1000)

    def test_non_3d_array_rejected_for_video(self):
        """Test that non-3D array is rejected for video.

        Spec: Input validation - Non-3D array rejected for video
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        video_2d = np.zeros((10, 8))  # 2D array

        with pytest.raises(ValueError):
            acq.set_simulated_video(video_2d, channel_name_video='cam', sample_rate=30)

    def test_channel_count_mismatch_rejected(self):
        """Test that channel count mismatch is rejected.

        Spec: Input validation - Channel count mismatch rejected
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        data = np.zeros((100, 3))  # 3 channels
        channel_names = ['ch0', 'ch1']  # Only 2 names

        with pytest.raises(ValueError):
            acq.set_simulated_data(data, channel_names=channel_names, sample_rate=1000)

    def test_invalid_fun_or_array_type_rejected(self):
        """Test that invalid fun_or_array type is rejected.

        Spec: Input validation - Invalid fun_or_array type rejected
        """
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test')

        with pytest.raises(ValueError):
            acq.set_simulated_data("not an array or function", channel_names=['ch0'], sample_rate=1000)


# =============================================================================
# Multiprocessing Disabled Test
# =============================================================================

class TestMultiprocessingDisabled:
    """Test that multiprocessing mode is explicitly disabled."""

    def test_multiprocessing_raises_value_error(self):
        """Test that multiprocessing raises ValueError.

        Spec: Multiprocessing mode disabled - Multiprocessing raises ValueError
        """
        with pytest.raises(ValueError, match="not supported"):
            LDAQ.simulator.SimulatedAcquisition(acquisition_name='test', multi_processing=True)


# =============================================================================
# Threading Mode Tests
# =============================================================================

class TestThreadingMode:
    """Tests for threading mode operation."""

    def test_data_generation_produces_samples(self, simple_simulated_acquisition):
        """Test that data generation produces samples.

        Spec: Threading mode operation - Data generation produces samples
        """
        acq = simple_simulated_acquisition  # 1kHz sample rate

        acq.run_acquisition(0.1)  # 100ms

        result = acq.get_measurement_dict()

        # Contract: approximately 100 samples at 1kHz for 100ms
        # Allow tolerance for timing variations
        assert result['data'].shape[0] >= 50  # At least half expected
        assert result['data'].shape[0] <= 200  # Not more than double expected

    def test_measurement_dictionary_contract(self, simple_simulated_acquisition, validate_measurement_dict):
        """Test that measurement dictionary contract is satisfied.

        Spec: Threading mode operation - Measurement dictionary contract satisfied
        """
        acq = simple_simulated_acquisition

        acq.run_acquisition(0.1)

        result = acq.get_measurement_dict()

        # Contract: must satisfy measurement dictionary contract
        validate_measurement_dict(result, source_level=True)

        # Additional checks
        assert 'time' in result
        assert 'data' in result
        assert 'channel_names' in result
        assert 'sample_rate' in result
        assert result['sample_rate'] == 1000
