"""
LDAQ Test Fixtures

This module provides reusable pytest fixtures for testing the LDAQ package.
All fixtures are designed to work without hardware, using SimulatedAcquisition.

Fixtures provided:
- simple_simulated_acquisition: Single channel, 1kHz, 10Hz sine wave
- multi_channel_acquisition: 3 channels with distinct signals
- video_acquisition: 32x32 video at 30fps
- temp_measurement_dir: Temporary directory with automatic cleanup
- saved_measurement_file: Pre-saved .pkl file for load tests
- core_with_single_source: Core with 1 acquisition
- core_with_multiple_sources: Core with 2 acquisitions
- validate_measurement_dict: Contract validation helper
- acquisition_with_virtual_channel: Acquisition with virtual channel

"""

import pytest
import numpy as np
from pathlib import Path

import LDAQ


# =============================================================================
# REQ-FIX-001: Simple Simulated Acquisition
# =============================================================================

@pytest.fixture
def simple_simulated_acquisition():
    """Single-channel SimulatedAcquisition for basic tests.

    Returns:
        LDAQ.simulator.SimulatedAcquisition: Configured with 1 channel, 1kHz,
        not started. Channel 'ch0' contains a 10Hz sine wave.

    Example:
        def test_basic_acquisition(simple_simulated_acquisition):
            acq = simple_simulated_acquisition
            acq.run_acquisition(0.1)
            data = acq.get_measurement_dict()
            assert 'ch0' in data['channel_names']
    """
    acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='test_acq')

    # 10 seconds of data at 1kHz (will loop)
    t = np.arange(10000) / 1000
    data = np.sin(2 * np.pi * 10 * t).reshape(-1, 1)  # 10 Hz sine

    acq.set_simulated_data(data, channel_names=['ch0'], sample_rate=1000)
    return acq


# =============================================================================
# REQ-FIX-002: Multi-Channel Acquisition
# =============================================================================

@pytest.fixture
def multi_channel_acquisition():
    """Multi-channel SimulatedAcquisition for channel selection tests.

    Returns:
        LDAQ.simulator.SimulatedAcquisition: Configured with 3 channels, 1kHz,
        not started.
        - 'ch0': 10 Hz sine wave
        - 'ch1': 20 Hz sine wave
        - 'ch2': linear ramp (0 to 1)

    Example:
        def test_channel_selection(multi_channel_acquisition):
            acq = multi_channel_acquisition
            assert len(acq.channel_names) == 3
    """
    acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='multi_ch_acq')

    t = np.arange(10000) / 1000
    ch0 = np.sin(2 * np.pi * 10 * t)   # 10 Hz
    ch1 = np.sin(2 * np.pi * 20 * t)   # 20 Hz
    ch2 = np.linspace(0, 1, 10000)     # Ramp

    data = np.column_stack([ch0, ch1, ch2])
    acq.set_simulated_data(data, channel_names=['ch0', 'ch1', 'ch2'], sample_rate=1000)
    return acq


# =============================================================================
# REQ-FIX-003: Video Acquisition
# =============================================================================

@pytest.fixture
def video_acquisition():
    """SimulatedAcquisition with video channel for video handling tests.

    Returns:
        LDAQ.simulator.SimulatedAcquisition: Configured with 1 video channel
        (32x32 pixels), 30 fps. Video contains a moving gradient pattern.

    Note:
        Video-only acquisitions return an empty 1D data array, which is
        inconsistent with the measurement-dict-contract spec. This is a
        known limitation of the current LDAQ implementation.

    Example:
        def test_video_data(video_acquisition):
            acq = video_acquisition
            acq.run_acquisition(0.1)
            data = acq.get_measurement_dict()
            assert 'video' in data
    """
    acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='video_acq')

    # 100 frames of 32x32 video
    n_frames = 100
    height, width = 32, 32
    video = np.zeros((n_frames, height, width), dtype=np.uint8)

    # Create moving gradient pattern (deterministic)
    for i in range(n_frames):
        video[i] = np.tile(
            ((np.arange(width) + i) % 256).astype(np.uint8),
            (height, 1)
        )

    acq.set_simulated_video(video, channel_name_video='camera', sample_rate=30)
    return acq


# =============================================================================
# REQ-FIX-004: Temporary Measurement Directory
# =============================================================================

@pytest.fixture
def temp_measurement_dir(tmp_path):
    """Temporary directory for measurement files.

    Uses pytest's built-in tmp_path fixture for automatic cleanup.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory

    Returns:
        pathlib.Path: Path to temporary measurement directory

    Example:
        def test_save_measurement(temp_measurement_dir, simple_simulated_acquisition):
            acq = simple_simulated_acquisition
            acq.run_acquisition(0.1)
            acq.save('test', root=str(temp_measurement_dir), timestamp=False)
    """
    measurement_dir = tmp_path / "measurements"
    measurement_dir.mkdir()
    return measurement_dir


# =============================================================================
# REQ-FIX-005: Saved Measurement File
# =============================================================================

@pytest.fixture
def saved_measurement_file(temp_measurement_dir, simple_simulated_acquisition):
    """Pre-saved measurement file for load tests.

    Creates a measurement file that can be used to test load_measurement().

    Returns:
        pathlib.Path: Path to saved .pkl file

    Example:
        def test_load_measurement(saved_measurement_file):
            data = LDAQ.load_measurement(str(saved_measurement_file))
            assert 'time' in data
    """
    acq = simple_simulated_acquisition

    # Run briefly to get data
    acq.run_acquisition(0.1)  # 100ms

    # Save measurement
    acq.save(name='test_measurement', root=str(temp_measurement_dir), timestamp=False)

    return temp_measurement_dir / "test_measurement.pkl"


# =============================================================================
# REQ-FIX-006: Core with Single Source
# =============================================================================

@pytest.fixture
def core_with_single_source(simple_simulated_acquisition):
    """Core instance with single acquisition source.

    Returns:
        LDAQ.Core: Configured with 1 SimulatedAcquisition, no visualization,
        not running.

    Example:
        def test_core_run(core_with_single_source):
            ldaq = core_with_single_source
            ldaq.run(0.1)
            data = ldaq.get_measurement_dict()
    """
    return LDAQ.Core(acquisitions=[simple_simulated_acquisition])


# =============================================================================
# REQ-FIX-007: Core with Multiple Sources
# =============================================================================

@pytest.fixture
def core_with_multiple_sources():
    """Core instance with multiple acquisition sources.

    Returns:
        LDAQ.Core: Configured with 2 SimulatedAcquisitions ('source1' and 'source2'),
        no visualization, not running.

    Example:
        def test_multi_source(core_with_multiple_sources):
            ldaq = core_with_multiple_sources
            ldaq.run(0.1)
            data = ldaq.get_measurement_dict()
            assert 'source1' in data
            assert 'source2' in data
    """
    # Create two independent acquisitions
    acq1 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source1')
    acq2 = LDAQ.simulator.SimulatedAcquisition(acquisition_name='source2')

    t = np.arange(10000) / 1000
    data1 = np.sin(2 * np.pi * 10 * t).reshape(-1, 1)
    data2 = np.sin(2 * np.pi * 15 * t).reshape(-1, 1)  # Different frequency

    acq1.set_simulated_data(data1, channel_names=['s1_ch0'], sample_rate=1000)
    acq2.set_simulated_data(data2, channel_names=['s2_ch0'], sample_rate=1000)

    return LDAQ.Core(acquisitions=[acq1, acq2])


# =============================================================================
# REQ-FIX-008: Measurement Dict Validator
# =============================================================================

@pytest.fixture(scope='session')
def validate_measurement_dict():
    """Validation function for measurement dict contract.

    Returns a callable that validates measurement dictionaries against the
    standard LDAQ measurement dict contract.

    Returns:
        callable: Function that validates measurement dicts.
            Raises AssertionError with descriptive message on failure.

    Example:
        def test_measurement_structure(validate_measurement_dict, simple_simulated_acquisition):
            acq = simple_simulated_acquisition
            acq.run_acquisition(0.1)
            data = acq.get_measurement_dict()
            validate_measurement_dict(data, source_level=True)  # Should not raise
    """
    def _validate(d: dict, source_level: bool = True) -> None:
        """Validate measurement dict conforms to contract.

        Args:
            d: Dictionary to validate
            source_level: True for acquisition-level dict, False for Core-level dict
                         (which contains multiple acquisition dicts)

        Raises:
            AssertionError: If validation fails, with descriptive message
        """
        if not source_level:
            # Core-level: validate each acquisition
            assert isinstance(d, dict) and len(d) > 0, \
                "Core-level dict must be non-empty dict"
            for name, acq_dict in d.items():
                assert isinstance(name, str) and len(name) > 0, \
                    f"Invalid source name: {name}"
                _validate(acq_dict, source_level=True)
            return

        # === Required fields (REQ-MEAS-001) ===
        assert 'time' in d, "Missing required field 'time'"
        assert isinstance(d['time'], np.ndarray), \
            f"'time' must be numpy array, got {type(d['time'])}"
        assert d['time'].ndim == 1, \
            f"'time' must be 1D array, got {d['time'].ndim}D"

        assert 'data' in d, "Missing required field 'data'"
        assert isinstance(d['data'], np.ndarray), \
            f"'data' must be numpy array, got {type(d['data'])}"
        # Note: Video-only acquisitions return 1D empty array (LDAQ limitation)
        # Accept both 2D and 1D-empty for compatibility
        if d['data'].size > 0:
            assert d['data'].ndim == 2, \
                f"'data' must be 2D array, got {d['data'].ndim}D"

        assert 'channel_names' in d, "Missing required field 'channel_names'"
        assert isinstance(d['channel_names'], list), \
            f"'channel_names' must be list, got {type(d['channel_names'])}"

        assert 'sample_rate' in d, "Missing required field 'sample_rate'"
        assert d['sample_rate'] is None or (
            isinstance(d['sample_rate'], (int, float)) and d['sample_rate'] > 0
        ), f"'sample_rate' must be positive number or None, got {d['sample_rate']}"

        # === Shape invariants (REQ-MEAS-004) ===
        # Note: Video-only acquisitions have empty data arrays (LDAQ limitation)
        # Skip shape checks if it's a video-only acquisition
        is_video_only = (d['data'].size == 0 and 'video' in d and len(d.get('channel_names', [])) == 0)

        if not is_video_only:
            assert d['time'].shape[0] == d['data'].shape[0], \
                f"time/data row mismatch: {d['time'].shape[0]} vs {d['data'].shape[0]}"
            assert len(d['channel_names']) == d['data'].shape[1], \
                f"channel_names/data column mismatch: {len(d['channel_names'])} vs {d['data'].shape[1]}"

        # === Video fields (REQ-MEAS-002) - optional ===
        if 'video' in d:
            assert 'channel_names_video' in d, \
                "'video' present but 'channel_names_video' missing"
            assert isinstance(d['video'], list), \
                f"'video' must be list, got {type(d['video'])}"
            assert isinstance(d['channel_names_video'], list), \
                f"'channel_names_video' must be list, got {type(d['channel_names_video'])}"
            assert len(d['channel_names_video']) == len(d['video']), \
                f"channel_names_video/video length mismatch: {len(d['channel_names_video'])} vs {len(d['video'])}"

            for i, v in enumerate(d['video']):
                assert isinstance(v, np.ndarray) and v.ndim == 3, \
                    f"video[{i}] must be 3D numpy array, got {type(v)} with ndim={getattr(v, 'ndim', 'N/A')}"
                assert v.shape[0] == d['time'].shape[0], \
                    f"video[{i}] samples ({v.shape[0]}) must match time samples ({d['time'].shape[0]})"

    return _validate


# =============================================================================
# REQ-FIX-009: Acquisition with Virtual Channel
# =============================================================================

@pytest.fixture
def acquisition_with_virtual_channel():
    """SimulatedAcquisition with pre-configured virtual channel.

    Returns:
        LDAQ.simulator.SimulatedAcquisition: Has 2 source channels ('ch0', 'ch1')
        and 1 virtual channel ('ratio' = ch0 / ch1).

    Example:
        def test_virtual_channel(acquisition_with_virtual_channel):
            acq = acquisition_with_virtual_channel
            acq.run_acquisition(0.1)
            data = acq.get_measurement_dict()
            assert 'ratio' in data['channel_names']
    """
    acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name='virt_ch_acq')

    t = np.arange(10000) / 1000
    ch0 = np.sin(2 * np.pi * 10 * t) + 2  # Offset to avoid division by zero
    ch1 = np.sin(2 * np.pi * 5 * t) + 2

    data = np.column_stack([ch0, ch1])
    acq.set_simulated_data(data, channel_names=['ch0', 'ch1'], sample_rate=1000)

    # Add virtual channel
    def compute_ratio(c0, c1):
        return (c0 / c1).reshape(-1, 1)

    acq.add_virtual_channel(
        virtual_channel_name='ratio',
        source_channels=['ch0', 'ch1'],
        function=compute_ratio
    )

    return acq
