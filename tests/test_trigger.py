"""
Tests for CustomPyTrigger class.

CustomPyTrigger extends pyTrigger to add sample count tracking, global trigger
synchronization, and continuous mode operation. These tests verify the ring buffer
management, triggering logic, and data retrieval mechanisms.

IMPORTANT: Tests use explicitly calculated expected values based on the contract,
NOT values read from the code. This ensures tests verify correctness, not just
internal consistency.

Contract Summary:
- Trigger 'up': fires at first sample >= trigger_level
- Trigger 'down': fires at first sample <= trigger_level
- Trigger 'abs': fires at first sample where |value| >= trigger_level
- When trigger fires at index T with presamples P (where T >= P):
  N_new_samples = P + (total_samples - T)
- get_data_new() returns N_new_samples rows, resets counter to 0
- get_data_new_PLOT() uses independent counter N_new_samples_PLOT

See LDAQ/acquisition_base.py lines 13-151 for implementation.
See openspec/specs/testing-standards/spec.md REQ-TEST-009 for testing requirements.
"""

from __future__ import annotations

import pytest
import numpy as np

from LDAQ.acquisition_base import CustomPyTrigger


# =============================================================================
# Constants for Test Calculations
# =============================================================================

# Helper to calculate trigger index for a linear ramp
def calc_trigger_index_up(start: float, end: float, n_samples: int, level: float) -> int:
    """Calculate the index where 'up' trigger fires for a linear ramp.

    Returns the first index where value >= level.
    For linspace(start, end, n_samples), value[i] = start + i * (end - start) / (n_samples - 1)
    """
    if end <= start:
        raise ValueError("For 'up' trigger, end must be > start")
    step = (end - start) / (n_samples - 1)
    # Solve: start + i * step >= level
    # i >= (level - start) / step
    i = int(np.ceil((level - start) / step))
    return min(i, n_samples - 1)


def calc_trigger_index_down(start: float, end: float, n_samples: int, level: float) -> int:
    """Calculate the index where 'down' trigger fires for a linear ramp.

    Returns the first index where value <= level.
    """
    if end >= start:
        raise ValueError("For 'down' trigger, end must be < start")
    step = (end - start) / (n_samples - 1)  # negative step
    # Solve: start + i * step <= level
    # i >= (level - start) / step  (note: step is negative, so inequality flips)
    i = int(np.ceil((level - start) / step))
    return min(i, n_samples - 1)


def calc_expected_n_new_samples(total_samples: int, trigger_index: int, presamples: int) -> int:
    """Calculate expected N_new_samples after trigger.

    Contract: N_new_samples = presamples + (total_samples - trigger_index)
    This assumes trigger_index >= presamples. For trigger_index < presamples,
    behavior is different (see test_presamples_insufficient).
    """
    return presamples + (total_samples - trigger_index)


# =============================================================================
# Module-Level Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_triggered_global():
    """Reset CustomPyTrigger.triggered_global after each test.

    The triggered_global class variable is shared across all instances and must
    be reset to False for test isolation. This is CRITICAL for test isolation
    per REQ-TEST-007.
    """
    yield
    CustomPyTrigger.triggered_global = False


# =============================================================================
# Helper Functions
# =============================================================================

def make_ramp_data(start: float, end: float, n_samples: int, n_channels: int) -> np.ndarray:
    """Create deterministic ramp data for trigger tests.

    Generates a linear ramp from start to end in the first channel (trigger channel),
    with other channels containing zeros. This makes trigger point calculation
    straightforward using linspace formula.

    Parameters
    ----------
    start : float
        Starting value of the ramp
    end : float
        Ending value of the ramp
    n_samples : int
        Number of samples to generate
    n_channels : int
        Number of channels in the output array

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_channels) with ramp in first channel
    """
    ramp = np.linspace(start, end, n_samples)
    data = np.zeros((n_samples, n_channels))
    data[:, 0] = ramp
    return data


# =============================================================================
# TestCustomPyTriggerInit - Initialization Tests
# =============================================================================

class TestCustomPyTriggerInit:
    """Test CustomPyTrigger initialization with various parameters."""

    def test_default_initialization(self):
        """Verify default initialization creates expected buffer and counters.

        Contract: Default values are rows=5120, channels=4, trigger_level=1.0,
        trigger_type='up', presamples=1000, all counters=0, triggered=False.
        """
        trigger = CustomPyTrigger()

        # Buffer dimensions - explicit expected values from __init__ signature
        assert trigger.ringbuff.rows == 5120
        assert trigger.ringbuff.columns == 4
        assert trigger.rows == 5120
        assert trigger.channels == 4

        # Sample counters - all must start at exactly 0
        assert trigger.N_acquired_samples == 0
        assert trigger.N_new_samples == 0
        assert trigger.N_acquired_samples_since_trigger == 0
        assert trigger.N_new_samples_PLOT == 0
        assert trigger.N_triggers == 0

        # State flags - initial state is untriggered
        assert trigger.triggered is False
        assert trigger.finished is False
        assert trigger.first_trigger is True

        # Trigger settings - explicit defaults from signature
        assert trigger.trigger_channel == 0
        assert trigger.trigger_level == 1.0
        assert trigger.trigger_type == 'up'
        assert trigger.presamples == 1000

    def test_custom_buffer_size(self):
        """Verify custom buffer dimensions are respected."""
        # Expected: buffer should have exactly the specified dimensions
        trigger = CustomPyTrigger(rows=500, channels=8)

        assert trigger.ringbuff.rows == 500
        assert trigger.ringbuff.columns == 8
        assert trigger.rows == 500
        assert trigger.channels == 8

    def test_custom_dtype(self):
        """Verify custom dtype is applied to ring buffer."""
        trigger = CustomPyTrigger(rows=100, channels=2, dtype=np.float32)

        # Contract: buffer data should have the specified dtype
        data = trigger.ringbuff.get_data()
        assert data.dtype == np.float32


# =============================================================================
# TestDataAddition - Data Addition and Counter Updates
# =============================================================================

class TestDataAddition:
    """Test data addition and sample counter updates."""

    def test_counter_updates_before_trigger(self):
        """Verify counters update correctly before trigger fires.

        Contract: Before trigger, N_acquired_samples increments by data length,
        but N_new_samples stays 0 because no triggered data exists yet.
        """
        # Setup: trigger level high enough that ramp won't reach it
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        # Expected values - calculated from contract
        assert trigger.N_acquired_samples == 100  # All samples added
        assert trigger.N_new_samples == 0  # No triggered data yet
        assert trigger.N_acquired_samples_since_trigger == 0  # No trigger
        assert trigger.triggered is False

    def test_counter_updates_after_trigger(self):
        """Verify counters update correctly after trigger fires.

        Contract: After trigger at index T with presamples P,
        N_new_samples = P + (total - T).
        """
        # Setup: trigger at 5.0 with 20 presamples
        presamples = 20
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=presamples)

        # Ramp 0→10 over 100 samples
        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)

        # Calculate expected trigger index: first sample >= 5.0
        trigger_idx = calc_trigger_index_up(0, 10, n_samples, 5.0)
        # Expected: index 50 (value = 5.05...)
        assert trigger_idx == 50, f"Test setup error: expected trigger at 50, got {trigger_idx}"

        trigger.add_data(data)

        # Calculate expected N_new_samples from contract
        expected_n_new = calc_expected_n_new_samples(n_samples, trigger_idx, presamples)
        # Expected: 20 + (100 - 50) = 70
        assert expected_n_new == 70, f"Test setup error: expected 70, got {expected_n_new}"

        # Verify against contract
        assert trigger.triggered is True
        assert trigger.N_new_samples == expected_n_new  # Exact value: 70
        assert trigger.N_acquired_samples == n_samples  # All 100 samples acquired

    def test_buffer_overflow(self):
        """Verify finished flag is set when buffer overflows.

        Contract: When data exceeds buffer capacity (rows_left), finished=True.
        """
        # Small buffer to easily overflow
        buffer_rows = 100
        trigger = CustomPyTrigger(rows=buffer_rows, channels=4, trigger_level=5.0,
                                 presamples=10)

        # Trigger first
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)
        assert trigger.triggered is True

        # Add more data than remaining buffer space
        # After first add, some rows are used. Adding 200 more will overflow.
        data2 = make_ramp_data(10, 20, 200, 4)
        trigger.add_data(data2)

        # Contract: finished should be True when buffer full
        assert trigger.finished is True


# =============================================================================
# TestTriggerDetection - Trigger Type Logic
# =============================================================================

class TestTriggerDetection:
    """Test trigger detection for different trigger types."""

    def test_trigger_type_up(self):
        """Verify 'up' trigger fires when crossing level upward.

        Contract: Trigger fires at first sample >= trigger_level.
        """
        trigger_level = 5.0
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=trigger_level,
                                 trigger_type='up', presamples=20)

        # Ramp from 0 to 10 - should cross 5.0 upward
        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)

        # Verify test data: trigger should fire at index 50
        trigger_idx = calc_trigger_index_up(0, 10, n_samples, trigger_level)
        assert trigger_idx == 50

        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_trigger_type_down(self):
        """Verify 'down' trigger fires when crossing level downward.

        Contract: Trigger fires at first sample <= trigger_level.
        """
        trigger_level = 5.0
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=trigger_level,
                                 trigger_type='down', presamples=20)

        # Ramp from 10 to 0 - should cross 5.0 downward
        n_samples = 100
        data = make_ramp_data(10, 0, n_samples, 4)

        # Verify test data: trigger should fire at index 50
        trigger_idx = calc_trigger_index_down(10, 0, n_samples, trigger_level)
        assert trigger_idx == 50

        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_trigger_type_abs(self):
        """Verify 'abs' trigger fires when absolute value exceeds level.

        Contract: Trigger fires when |value| >= trigger_level.
        """
        trigger_level = 5.0
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=trigger_level,
                                 trigger_type='abs', presamples=20)

        # Ramp from -10 to 0: values start at -10 (|−10| = 10 > 5), trigger immediately
        # Actually, trigger should fire at first sample since |-10| >= 5
        data = make_ramp_data(-10, 0, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_no_trigger(self):
        """Verify trigger does not fire when level is never crossed.

        Contract: triggered remains False if no sample meets trigger condition.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                 trigger_type='up', presamples=10)

        # Ramp stays well below trigger level (max value = 10, trigger = 100)
        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is False
        assert trigger.N_triggers == 0


# =============================================================================
# TestPresamples - Presample Handling
# =============================================================================

class TestPresamples:
    """Test presample inclusion in triggered data."""

    def test_presamples_included(self):
        """Verify presamples are included in triggered data count.

        Contract: N_new_samples = presamples + (total - trigger_index)
        when trigger_index >= presamples.
        """
        presamples = 20
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=presamples)

        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)

        # Trigger fires at index 50 (first value >= 5.0)
        trigger_idx = calc_trigger_index_up(0, 10, n_samples, 5.0)
        assert trigger_idx == 50  # Verify test assumption
        assert trigger_idx >= presamples  # Ensure we're testing the normal case

        trigger.add_data(data)

        # Expected: presamples + post-trigger = 20 + 50 = 70
        expected_n_new = calc_expected_n_new_samples(n_samples, trigger_idx, presamples)
        assert expected_n_new == 70  # Verify calculation

        assert trigger.triggered is True
        assert trigger.N_new_samples == expected_n_new

    def test_presamples_at_boundary(self):
        """Test when trigger_index exactly equals presamples.

        Contract: N_new_samples = presamples + (total - trigger_index)
        At boundary, this is a well-defined case.
        """
        presamples = 20
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=2.0,
                                 trigger_type='up', presamples=presamples)

        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)

        # Trigger level 2.0 in ramp 0→10 over 100 samples
        # value[i] = 10 * i / 99, solve: 10i/99 >= 2 → i >= 19.8 → i = 20
        trigger_idx = calc_trigger_index_up(0, 10, n_samples, 2.0)
        assert trigger_idx == 20  # Exactly at presamples boundary

        trigger.add_data(data)

        # Expected: 20 + (100 - 20) = 100
        expected_n_new = calc_expected_n_new_samples(n_samples, trigger_idx, presamples)
        assert expected_n_new == 100

        assert trigger.triggered is True
        assert trigger.N_new_samples == expected_n_new


# =============================================================================
# TestDataRetrieval - Data Retrieval Methods
# =============================================================================

class TestDataRetrieval:
    """Test data retrieval methods (get_data_new, get_data_new_PLOT)."""

    def test_get_data_new_returns_correct_shape(self):
        """Verify get_data_new returns array with expected dimensions.

        Contract: Returns (N_new_samples, channels) array.
        """
        presamples = 20
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=presamples)

        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)
        trigger.add_data(data)

        # Calculate expected shape
        trigger_idx = calc_trigger_index_up(0, 10, n_samples, 5.0)
        expected_rows = calc_expected_n_new_samples(n_samples, trigger_idx, presamples)
        # Expected: 70 rows, 4 columns

        retrieved = trigger.get_data_new()

        assert retrieved.shape == (expected_rows, 4)

    def test_get_data_new_resets_counter(self):
        """Verify get_data_new resets N_new_samples to 0.

        Contract: After retrieval, N_new_samples = 0.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=20)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.N_new_samples > 0  # Sanity check

        trigger.get_data_new()

        assert trigger.N_new_samples == 0  # Contract: counter reset

    def test_get_data_new_subsequent_returns_only_new(self):
        """Verify subsequent get_data_new calls return only new data.

        Contract: Second call returns only data added since first retrieval.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=20)

        # Trigger
        data1 = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data1)

        # First retrieval clears counter
        trigger.get_data_new()
        assert trigger.N_new_samples == 0

        # Add exactly 50 new samples
        n_new_samples = 50
        data2 = make_ramp_data(10, 20, n_new_samples, 4)
        trigger.add_data(data2)

        # Second retrieval should return exactly 50 samples
        second_retrieval = trigger.get_data_new()
        assert second_retrieval.shape[0] == n_new_samples
        assert trigger.N_new_samples == 0

    def test_get_data_new_empty_when_no_new_data(self):
        """Verify get_data_new returns empty array when no new data.

        Contract: Returns (0, channels) array when N_new_samples = 0.
        """
        n_channels = 4
        trigger = CustomPyTrigger(rows=500, channels=n_channels, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        data = make_ramp_data(0, 10, 100, n_channels)
        trigger.add_data(data)

        # First call retrieves data
        trigger.get_data_new()

        # Second call without new data
        empty = trigger.get_data_new()
        assert empty.shape == (0, n_channels)

    def test_get_data_new_before_trigger_returns_empty(self):
        """Verify get_data_new before trigger returns empty array.

        Contract: Before trigger, returns (0, channels) array.
        """
        n_channels = 4
        trigger = CustomPyTrigger(rows=500, channels=n_channels, trigger_level=100.0)

        # Add data but don't trigger (level too high)
        data = make_ramp_data(0, 10, 100, n_channels)
        trigger.add_data(data)

        assert trigger.triggered is False
        empty = trigger.get_data_new()
        assert empty.shape == (0, n_channels)

    def test_get_data_new_plot_independent_counter(self):
        """Verify N_new_samples_PLOT is independent of N_new_samples.

        Contract: get_data_new does not affect N_new_samples_PLOT.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=20)

        n_samples = 100
        data = make_ramp_data(0, 10, n_samples, 4)
        trigger.add_data(data)

        # N_new_samples_PLOT should equal N_acquired_samples (all data added)
        assert trigger.N_new_samples_PLOT == n_samples

        # Get data using normal retrieval
        trigger.get_data_new()

        # N_new_samples_PLOT should be unchanged
        assert trigger.N_new_samples_PLOT == n_samples  # Independent counter

    def test_get_data_new_plot_works_before_trigger(self):
        """Verify get_data_new_PLOT works before trigger fires.

        Contract: get_data_new_PLOT returns data based on N_new_samples_PLOT,
        which tracks all data added, not just triggered data.
        """
        n_channels = 4
        n_samples = 100
        trigger = CustomPyTrigger(rows=500, channels=n_channels, trigger_level=100.0)

        data = make_ramp_data(0, 10, n_samples, n_channels)
        trigger.add_data(data)

        assert trigger.triggered is False  # No trigger
        plot_data = trigger.get_data_new_PLOT()

        # Should return all samples added
        assert plot_data.shape == (n_samples, n_channels)
        assert trigger.N_new_samples_PLOT == 0  # Counter reset after retrieval


# =============================================================================
# TestContinuousMode - Continuous Acquisition Mode
# =============================================================================

class TestContinuousMode:
    """Test continuous mode operation and buffer reset."""

    def test_continuous_mode_resets_buffer_when_full(self):
        """Verify continuous mode resets buffer when full instead of stopping.

        Contract: In continuous mode with no duration limit, buffer resets
        when full and finished stays False.
        """
        buffer_rows = 100
        trigger = CustomPyTrigger(rows=buffer_rows, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = None  # No limit

        # Trigger first
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)
        assert trigger.triggered is True

        # Add data that will exceed buffer, forcing reset
        data2 = make_ramp_data(10, 20, 80, 4)  # More than remaining space
        trigger.add_data(data2)

        # Contract: continuous mode should reset buffer, not finish
        assert trigger.finished is False
        # After reset and adding data2, rows_left should be buffer_rows - len(data2)
        assert trigger.rows_left == buffer_rows - len(data2)

    def test_continuous_mode_respects_duration_limit(self):
        """Verify continuous mode respects N_samples_to_acquire limit.

        Contract: When N_samples_to_acquire is set, finished=True when limit reached.
        """
        sample_limit = 100
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = sample_limit

        # Trigger
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)

        # Add data to exceed limit
        data2 = make_ramp_data(10, 20, 150, 4)
        trigger.add_data(data2)

        # Contract: finished=True when limit reached
        assert trigger.finished is True
        assert trigger.N_acquired_samples_since_trigger <= sample_limit

    def test_continuous_mode_no_limit_continues_indefinitely(self):
        """Verify continuous mode without limit continues past buffer size.

        Contract: With N_samples_to_acquire=None, acquisition doesn't stop.
        """
        buffer_rows = 100
        trigger = CustomPyTrigger(rows=buffer_rows, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = None

        # Trigger
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)

        # Add much more data than buffer size (multiple resets)
        total_added = 50
        for _ in range(5):
            data = make_ramp_data(10, 20, 50, 4)
            trigger.add_data(data)
            total_added += 50

        # Contract: finished stays False with no limit
        assert trigger.finished is False
        assert trigger.N_acquired_samples == total_added


# =============================================================================
# TestGlobalTriggerSync - Global Trigger Synchronization
# =============================================================================

class TestGlobalTriggerSync:
    """Test global trigger synchronization across instances."""

    def test_first_instance_sets_triggered_global(self):
        """Verify first instance to trigger sets the class variable.

        Contract: When instance triggers, CustomPyTrigger.triggered_global = True.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)

        # Precondition
        assert CustomPyTrigger.triggered_global is False

        data = make_ramp_data(0, 10, 100, 4)
        trigger_a.add_data(data)

        # Contract: both instance and class variable should be True
        assert trigger_a.triggered is True
        assert CustomPyTrigger.triggered_global is True

    def test_second_instance_receives_global_trigger(self):
        """Verify second instance triggers via triggered_global.

        Contract: Instance B triggers when adding data if triggered_global is True,
        even if its own trigger condition isn't met.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)
        # Instance B has impossible trigger level
        trigger_b = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                   presamples=10)

        # Trigger instance A
        data_a = make_ramp_data(0, 10, 100, 4)
        trigger_a.add_data(data_a)

        assert trigger_a.triggered is True
        assert CustomPyTrigger.triggered_global is True

        # Add data to instance B (won't self-trigger, level too high)
        data_b = make_ramp_data(0, 10, 100, 4)
        trigger_b.add_data(data_b)

        # Contract: B triggers via global sync
        assert trigger_b.triggered is True

    def test_both_instances_count_trigger_once(self):
        """Verify both instances increment N_triggers exactly once.

        Contract: N_triggers = 1 for both after global sync.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)
        trigger_b = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                   presamples=10)

        # Trigger instance A
        trigger_a.add_data(make_ramp_data(0, 10, 100, 4))

        # Trigger instance B via global
        trigger_b.add_data(make_ramp_data(0, 10, 100, 4))

        # Contract: each instance counts trigger exactly once
        assert trigger_a.N_triggers == 1
        assert trigger_b.N_triggers == 1


# =============================================================================
# TestTriggerCount - Trigger Counting Logic
# =============================================================================

class TestTriggerCount:
    """Test N_triggers counter and first_trigger flag."""

    def test_single_trigger_event_counted_once(self):
        """Verify N_triggers = 1 after trigger and first_trigger = False.

        Contract: Trigger event increments N_triggers once and clears first_trigger.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)

        # Preconditions from contract
        assert trigger.first_trigger is True
        assert trigger.N_triggers == 0

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        # Contract: exactly one trigger counted
        assert trigger.N_triggers == 1
        assert trigger.first_trigger is False

    def test_no_double_counting_on_subsequent_data(self):
        """Verify N_triggers doesn't increment on subsequent data additions.

        Contract: After first trigger, N_triggers stays at 1 regardless of
        how much more data is added.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)

        # First trigger
        trigger.add_data(make_ramp_data(0, 10, 100, 4))
        assert trigger.N_triggers == 1

        # Add more data multiple times
        for _ in range(3):
            trigger.add_data(make_ramp_data(10, 20, 50, 4))
            assert trigger.N_triggers == 1  # Must stay at 1
