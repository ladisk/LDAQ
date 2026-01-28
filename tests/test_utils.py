"""
Tests for LDAQ utility functions.

This module tests the save/load functionality in utils.py:
- load_measurement() - single file loading
- load_measurement_multiple_files() - multi-file concatenation
- load_measurement_multiple_files_memmap() - memory-mapped loading

See openspec/changes/test-utils/specs/ for detailed requirements.
"""

import pytest
import numpy as np
import pickle
import os

import LDAQ


# =============================================================================
# Single File Loading Tests
# =============================================================================

class TestSingleFileLoading:
    """Tests for load_measurement() function."""

    def test_load_valid_measurement_file(self, saved_measurement_file):
        """Test loading a valid measurement file.

        Spec: Single file loading - Load valid measurement file
        """
        result = LDAQ.load_measurement(str(saved_measurement_file))

        # Contract: returns dict with measurement data
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'data' in result
        assert 'channel_names' in result

    def test_load_with_directory_parameter(self, temp_measurement_dir, simple_simulated_acquisition):
        """Test loading with separate directory parameter.

        Spec: Single file loading - Load with directory parameter
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)
        acq.save(name='test_file', root=str(temp_measurement_dir), timestamp=False)

        result = LDAQ.load_measurement('test_file.pkl', directory=str(temp_measurement_dir))

        # Contract: loads from directory/name
        assert isinstance(result, dict)
        assert 'time' in result

    def test_file_not_found_raises_error(self, temp_measurement_dir):
        """Test that file not found raises FileNotFoundError.

        Spec: Single file loading - File not found raises error
        """
        with pytest.raises(FileNotFoundError):
            LDAQ.load_measurement('nonexistent_file.pkl', directory=str(temp_measurement_dir))


# =============================================================================
# Multi-File Concatenation Tests
# =============================================================================

class TestMultiFileConcatenation:
    """Tests for load_measurement_multiple_files() function."""

    @pytest.fixture
    def multi_file_dir(self, temp_measurement_dir):
        """Create directory with multiple measurement files."""
        # Create mock Core-level measurement dicts
        for i in range(3):
            measurement = {
                'source1': {
                    'time': np.arange(10) + i * 10,  # 0-9, 10-19, 20-29
                    'data': np.ones((10, 2)) * (i + 1),  # values 1, 2, 3
                    'channel_names': ['ch0', 'ch1'],
                    'sample_rate': 1000
                }
            }
            filepath = temp_measurement_dir / f'test_meas_{i}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(measurement, f)

        return temp_measurement_dir

    def test_data_arrays_concatenated(self, multi_file_dir):
        """Test that data arrays are concatenated along axis 0.

        Spec: Multi-file concatenation - Data arrays concatenated
        """
        result = LDAQ.load_measurement_multiple_files(
            directory=str(multi_file_dir),
            contains='test_meas'
        )

        # Contract: data concatenated from 3 files, each with 10 rows = 30 total
        assert result['source1']['data'].shape[0] == 30
        assert result['source1']['data'].shape[1] == 2

    def test_time_arrays_concatenated(self, multi_file_dir):
        """Test that time arrays are concatenated along axis 0.

        Spec: Multi-file concatenation - Time arrays concatenated
        """
        result = LDAQ.load_measurement_multiple_files(
            directory=str(multi_file_dir),
            contains='test_meas'
        )

        # Contract: time concatenated from 3 files, each with 10 rows = 30 total
        assert result['source1']['time'].shape[0] == 30

    def test_contains_filter_works(self, multi_file_dir):
        """Test that contains filter works.

        Spec: Multi-file concatenation - Contains filter works
        """
        # Create a file that shouldn't match
        other_file = multi_file_dir / 'other_file.pkl'
        with open(other_file, 'wb') as f:
            pickle.dump({'other': {'data': np.zeros((5, 1))}}, f)

        result = LDAQ.load_measurement_multiple_files(
            directory=str(multi_file_dir),
            contains='test_meas'
        )

        # Contract: only test_meas files loaded, not other_file
        assert 'source1' in result
        assert 'other' not in result

    def test_no_matching_files_returns_empty(self, temp_measurement_dir):
        """Test that no matching files returns empty dict.

        Spec: Multi-file concatenation - No matching files returns empty
        """
        result = LDAQ.load_measurement_multiple_files(
            directory=str(temp_measurement_dir),
            contains='nonexistent_pattern'
        )

        # Contract: empty dict when no files match
        assert result == {}


# =============================================================================
# Memory-Mapped Loading Tests
# =============================================================================

class TestMemoryMappedLoading:
    """Tests for load_measurement_multiple_files_memmap() function."""

    @pytest.fixture
    def multi_file_dir_for_memmap(self, temp_measurement_dir):
        """Create directory with multiple measurement files for memmap testing."""
        for i in range(2):
            measurement = {
                'source1': {
                    'time': np.arange(10, dtype=np.float64) + i * 10,
                    'data': np.ones((10, 2), dtype=np.float64) * (i + 1),
                    'channel_names': ['ch0', 'ch1'],
                    'sample_rate': 1000
                }
            }
            filepath = temp_measurement_dir / f'memmap_test_{i}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(measurement, f)

        return temp_measurement_dir

    def test_returns_memmap_arrays(self, multi_file_dir_for_memmap, tmp_path):
        """Test that returns memmap arrays.

        Spec: Memory-mapped loading - Returns memmap arrays
        """
        tmp_dir = tmp_path / 'memmap_tmp'

        result = LDAQ.load_measurement_multiple_files_memmap(
            directory=str(multi_file_dir_for_memmap),
            contains='memmap_test',
            tmp_dir=str(tmp_dir)
        )

        # Contract: data and time are memmap objects
        assert isinstance(result['source1']['data'], np.memmap)
        assert isinstance(result['source1']['time'], np.memmap)

    def test_creates_temp_directory(self, multi_file_dir_for_memmap, tmp_path):
        """Test that creates temp directory if it doesn't exist.

        Spec: Memory-mapped loading - Creates temp directory
        """
        tmp_dir = tmp_path / 'new_memmap_dir'
        assert not tmp_dir.exists()

        LDAQ.load_measurement_multiple_files_memmap(
            directory=str(multi_file_dir_for_memmap),
            contains='memmap_test',
            tmp_dir=str(tmp_dir)
        )

        # Contract: tmp_dir should be created
        assert tmp_dir.exists()

    def test_no_matching_files_raises_error(self, temp_measurement_dir, tmp_path):
        """Test that no matching files raises ValueError.

        Spec: Memory-mapped loading - No matching files raises error
        """
        tmp_dir = tmp_path / 'memmap_tmp'

        with pytest.raises(ValueError, match="No matching files"):
            LDAQ.load_measurement_multiple_files_memmap(
                directory=str(temp_measurement_dir),
                contains='nonexistent_pattern',
                tmp_dir=str(tmp_dir)
            )


# =============================================================================
# Round-Trip Integrity Tests
# =============================================================================

class TestRoundTripIntegrity:
    """Tests for save → load round-trip integrity."""

    def test_data_values_preserved(self, simple_simulated_acquisition, temp_measurement_dir):
        """Test that data values are preserved after save/load.

        Spec: Round-trip integrity - Data values preserved
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        original = acq.get_measurement_dict()
        acq.save(name='roundtrip_test', root=str(temp_measurement_dir), timestamp=False)

        loaded = LDAQ.load_measurement('roundtrip_test.pkl', directory=str(temp_measurement_dir))

        # Contract: data values should be identical
        np.testing.assert_array_equal(loaded['data'], original['data'])
        np.testing.assert_array_equal(loaded['time'], original['time'])

    def test_channel_names_preserved(self, simple_simulated_acquisition, temp_measurement_dir):
        """Test that channel names are preserved after save/load.

        Spec: Round-trip integrity - Channel names preserved
        """
        acq = simple_simulated_acquisition
        acq.run_acquisition(0.1)

        original = acq.get_measurement_dict()
        acq.save(name='roundtrip_test2', root=str(temp_measurement_dir), timestamp=False)

        loaded = LDAQ.load_measurement('roundtrip_test2.pkl', directory=str(temp_measurement_dir))

        # Contract: channel names should be identical
        assert loaded['channel_names'] == original['channel_names']
        assert loaded['sample_rate'] == original['sample_rate']
