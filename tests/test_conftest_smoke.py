"""
Smoke tests for conftest.py fixtures.

These tests verify that the fixtures are functional and return correct types.
They are intentionally minimal - detailed testing is done in other test files.
"""

import pytest
import numpy as np
from pathlib import Path

import LDAQ


class TestFixtureSmoke:
    """Smoke tests for all conftest fixtures."""

    def test_simple_simulated_acquisition(self, simple_simulated_acquisition):
        """Verify simple_simulated_acquisition fixture."""
        acq = simple_simulated_acquisition

        assert acq is not None
        assert isinstance(acq, LDAQ.simulator.SimulatedAcquisition)
        assert acq.acquisition_name == 'test_acq'
        assert acq.sample_rate == 1000
        assert len(acq.channel_names) == 1
        assert acq.channel_names[0] == 'ch0'

    def test_multi_channel_acquisition(self, multi_channel_acquisition):
        """Verify multi_channel_acquisition fixture."""
        acq = multi_channel_acquisition

        assert acq is not None
        assert isinstance(acq, LDAQ.simulator.SimulatedAcquisition)
        assert acq.acquisition_name == 'multi_ch_acq'
        assert acq.sample_rate == 1000
        assert len(acq.channel_names) == 3
        assert acq.channel_names == ['ch0', 'ch1', 'ch2']

    def test_video_acquisition(self, video_acquisition):
        """Verify video_acquisition fixture."""
        acq = video_acquisition

        assert acq is not None
        assert isinstance(acq, LDAQ.simulator.SimulatedAcquisition)
        assert acq.acquisition_name == 'video_acq'
        assert acq.sample_rate == 30
        assert len(acq.channel_names_video) == 1
        assert acq.channel_names_video[0] == 'camera'
        # Video-only acquisition (no data channels)
        # Note: SimulatedAcquisition cannot have both data and video channels
        # simultaneously due to implementation limitations.

    def test_temp_measurement_dir(self, temp_measurement_dir):
        """Verify temp_measurement_dir fixture."""
        assert temp_measurement_dir is not None
        assert isinstance(temp_measurement_dir, Path)
        assert temp_measurement_dir.exists()
        assert temp_measurement_dir.is_dir()

    def test_saved_measurement_file(self, saved_measurement_file):
        """Verify saved_measurement_file fixture."""
        assert saved_measurement_file is not None
        assert isinstance(saved_measurement_file, Path)
        assert saved_measurement_file.exists()
        assert saved_measurement_file.suffix == '.pkl'

        # Verify it's loadable
        data = LDAQ.load_measurement(str(saved_measurement_file))
        assert 'time' in data
        assert 'data' in data
        assert 'channel_names' in data

    def test_core_with_single_source(self, core_with_single_source):
        """Verify core_with_single_source fixture."""
        ldaq = core_with_single_source

        assert ldaq is not None
        assert isinstance(ldaq, LDAQ.Core)
        assert len(ldaq.acquisitions) == 1

    def test_core_with_multiple_sources(self, core_with_multiple_sources):
        """Verify core_with_multiple_sources fixture."""
        ldaq = core_with_multiple_sources

        assert ldaq is not None
        assert isinstance(ldaq, LDAQ.Core)
        assert len(ldaq.acquisitions) == 2

        names = [acq.acquisition_name for acq in ldaq.acquisitions]
        assert 'source1' in names
        assert 'source2' in names

    def test_validate_measurement_dict_valid(self, validate_measurement_dict, simple_simulated_acquisition):
        """Verify validate_measurement_dict accepts valid data."""
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)
        data = acq.get_measurement_dict()

        # Should not raise
        validate_measurement_dict(data, source_level=True)

    def test_validate_measurement_dict_rejects_invalid(self, validate_measurement_dict):
        """Verify validate_measurement_dict rejects invalid data."""
        # Missing required field
        with pytest.raises(AssertionError, match="Missing required field 'time'"):
            validate_measurement_dict({'data': np.array([[1, 2]])}, source_level=True)

        # Wrong shape
        with pytest.raises(AssertionError, match="must be 1D"):
            validate_measurement_dict({
                'time': np.array([[1, 2]]),  # 2D instead of 1D
                'data': np.array([[1, 2]]),
                'channel_names': ['ch0', 'ch1'],
                'sample_rate': 1000
            }, source_level=True)

    def test_acquisition_with_virtual_channel(self, acquisition_with_virtual_channel):
        """Verify acquisition_with_virtual_channel fixture."""
        acq = acquisition_with_virtual_channel

        assert acq is not None
        assert isinstance(acq, LDAQ.simulator.SimulatedAcquisition)
        assert acq.acquisition_name == 'virt_ch_acq'

        # Check base channels
        assert 'ch0' in acq.channel_names
        assert 'ch1' in acq.channel_names

        # Check virtual channel exists
        assert 'ratio' in acq.channel_names


class TestFixtureIntegration:
    """Integration tests verifying fixtures work together."""

    def test_run_acquisition_and_validate(self, simple_simulated_acquisition, validate_measurement_dict):
        """Test that acquired data passes validation."""
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)
        data = acq.get_measurement_dict()

        # Validate
        validate_measurement_dict(data, source_level=True)

        # Check data shape
        assert data['time'].shape[0] > 0
        assert data['data'].shape[0] == data['time'].shape[0]
        assert data['data'].shape[1] == 1

    def test_core_run_and_validate(self, simple_simulated_acquisition, validate_measurement_dict):
        """Test that Core-level data passes validation.

        Note: Uses run_acquisition() directly because Core.run() requires
        keyboard hotkeys which need root on Linux.
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        # Create Core-level dict manually (simulating what Core.get_measurement_dict returns)
        data = {acq.acquisition_name: acq.get_measurement_dict()}

        # Core-level validation (contains acquisition dicts)
        validate_measurement_dict(data, source_level=False)

    def test_video_acquisition_and_validate(self, video_acquisition, validate_measurement_dict):
        """Test that video data passes validation."""
        acq = video_acquisition
        acq.run_acquisition(0.1)
        data = acq.get_measurement_dict()

        # Validate including video fields
        validate_measurement_dict(data, source_level=True)

        # Check video data
        assert 'video' in data
        assert len(data['video']) == 1
        assert data['video'][0].ndim == 3
