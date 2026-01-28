"""
Tests for CustomPyTrigger class.

CustomPyTrigger extends pyTrigger to add sample count tracking, global trigger
synchronization, and continuous mode operation. These tests verify the ring buffer
management, triggering logic, and data retrieval mechanisms.

See LDAQ/acquisition_base.py lines 13-151 for implementation.
"""

from __future__ import annotations

import pytest
import numpy as np

from LDAQ.acquisition_base import CustomPyTrigger


# =============================================================================
# Module-Level Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_triggered_global():
    """Reset CustomPyTrigger.triggered_global after each test.

    The triggered_global class variable is shared across all instances and must
    be reset to False for test isolation.

    Yields:
        None: Control returns to test, then cleanup runs
    """
    yield
    CustomPyTrigger.triggered_global = False


# =============================================================================
# Helper Functions
# =============================================================================

def make_ramp_data(start: float, end: float, n_samples: int, n_channels: int) -> np.ndarray:
    """Create deterministic ramp data for trigger tests.

    Generates a linear ramp from start to end in the first channel, with other
    channels containing zeros. This makes trigger point prediction straightforward.

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

    Examples
    --------
    >>> data = make_ramp_data(0, 10, 100, 4)
    >>> data.shape
    (100, 4)
    >>> data[0, 0], data[-1, 0]
    (0.0, 10.0)
    >>> np.all(data[:, 1:] == 0)
    True
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

        Default should be 5120 rows, 4 channels, all counters at zero.
        """
        trigger = CustomPyTrigger()

        # Buffer dimensions
        assert trigger.ringbuff.rows == 5120
        assert trigger.ringbuff.columns == 4
        assert trigger.rows == 5120
        assert trigger.channels == 4

        # Sample counters
        assert trigger.N_acquired_samples == 0
        assert trigger.N_new_samples == 0
        assert trigger.N_acquired_samples_since_trigger == 0
        assert trigger.N_new_samples_PLOT == 0
        assert trigger.N_triggers == 0

        # State flags
        assert trigger.triggered is False
        assert trigger.finished is False
        assert trigger.first_trigger is True

        # Trigger settings
        assert trigger.trigger_channel == 0
        assert trigger.trigger_level == 1.0
        assert trigger.trigger_type == 'up'
        assert trigger.presamples == 1000

    def test_custom_buffer_size(self):
        """Verify custom buffer dimensions are respected."""
        trigger = CustomPyTrigger(rows=500, channels=8)

        assert trigger.ringbuff.rows == 500
        assert trigger.ringbuff.columns == 8
        assert trigger.rows == 500
        assert trigger.channels == 8

    def test_custom_dtype(self):
        """Verify custom dtype is applied to ring buffer."""
        trigger = CustomPyTrigger(rows=100, channels=2, dtype=np.float32)

        # Check dtype through get_data() since RingBuffer2D doesn't expose dtype directly
        data = trigger.ringbuff.get_data()
        assert data.dtype == np.float32


# =============================================================================
# TestDataAddition - Data Addition and Counter Updates
# =============================================================================

class TestDataAddition:
    """Test data addition and sample counter updates."""

    def test_counter_updates_before_trigger(self):
        """Verify counters update correctly before trigger fires.

        Before trigger: N_acquired_samples increments, but N_new_samples stays 0
        because no triggered data is available yet.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.N_acquired_samples == 100
        assert trigger.N_new_samples == 0  # No triggered data yet
        assert trigger.N_acquired_samples_since_trigger == 0
        assert trigger.triggered is False

    def test_counter_updates_after_trigger(self):
        """Verify counters update correctly after trigger fires.

        After trigger: Both N_acquired_samples and N_new_samples should increment.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        # First chunk: trigger will fire when crossing 5.0
        data1 = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data1)

        assert trigger.triggered is True
        n_new_after_trigger = trigger.N_new_samples
        assert n_new_after_trigger > 0

        # Second chunk: should increment counters
        data2 = make_ramp_data(10, 20, 50, 4)
        trigger.add_data(data2)

        assert trigger.N_acquired_samples == 150
        assert trigger.N_new_samples == n_new_after_trigger + 50
        assert trigger.N_acquired_samples_since_trigger == n_new_after_trigger + 50

    def test_buffer_overflow(self):
        """Verify finished flag is set when buffer overflows.

        When data exceeds buffer capacity without continuous mode, finished=True.
        """
        trigger = CustomPyTrigger(rows=100, channels=4, trigger_level=5.0,
                                 presamples=10)

        # Trigger first
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)
        assert trigger.triggered is True

        # Add more data than buffer can hold
        data2 = make_ramp_data(10, 20, 200, 4)
        trigger.add_data(data2)

        assert trigger.finished is True


# =============================================================================
# TestTriggerDetection - Trigger Type Logic
# =============================================================================

class TestTriggerDetection:
    """Test trigger detection for different trigger types."""

    def test_trigger_type_up(self):
        """Verify 'up' trigger fires when crossing level upward."""
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        # Ramp from 0 to 10 - should cross 5.0 upward
        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_trigger_type_down(self):
        """Verify 'down' trigger fires when crossing level downward."""
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='down', presamples=10)

        # Ramp from 10 to 0 - should cross 5.0 downward
        data = make_ramp_data(10, 0, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_trigger_type_abs(self):
        """Verify 'abs' trigger fires when absolute value exceeds level.

        Should trigger when |value| > trigger_level, regardless of sign.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='abs', presamples=10)

        # Ramp from -10 to 0 - absolute value should exceed 5.0
        data = make_ramp_data(-10, 0, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        assert trigger.N_triggers == 1

    def test_no_trigger(self):
        """Verify trigger does not fire when level is never crossed."""
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                 trigger_type='up', presamples=10)

        # Ramp stays well below trigger level
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
        """Verify presamples are included in triggered data.

        When trigger fires, N_new_samples should include presamples.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=50)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        # Trigger fires somewhere around sample 50 (midpoint of 0-10 ramp crossing 5.0)
        # N_new_samples should include presamples + post-trigger samples
        assert trigger.N_new_samples >= 50  # At least presamples

    def test_presamples_insufficient(self):
        """Verify no error when presamples exceed available data.

        If trigger fires early, use whatever presamples are available.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=1.0,
                                 trigger_type='up', presamples=100)

        # Trigger fires very early (crossing 1.0 at ~10% of ramp)
        data = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        # Should not raise error, just use available presamples


# =============================================================================
# TestDataRetrieval - Data Retrieval Methods
# =============================================================================

class TestDataRetrieval:
    """Test data retrieval methods (get_data_new, get_data_new_PLOT)."""

    def test_get_data_new_first_retrieval(self):
        """Verify first get_data_new retrieval after trigger returns correct data.

        Should return all data since trigger and reset N_new_samples to 0.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is True
        n_new_before = trigger.N_new_samples
        assert n_new_before > 0

        # Get new data
        retrieved = trigger.get_data_new()

        assert retrieved.shape[0] == n_new_before
        assert retrieved.shape[1] == 4
        assert trigger.N_new_samples == 0  # Reset after retrieval

    def test_get_data_new_subsequent(self):
        """Verify subsequent get_data_new calls return only new data.

        Second call should return only data added since first retrieval.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        # Trigger
        data1 = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data1)

        # First retrieval
        first_retrieval = trigger.get_data_new()
        assert trigger.N_new_samples == 0

        # Add more data
        data2 = make_ramp_data(10, 20, 50, 4)
        trigger.add_data(data2)

        # Second retrieval should return only the 50 new samples
        second_retrieval = trigger.get_data_new()
        assert second_retrieval.shape[0] == 50
        assert trigger.N_new_samples == 0

    def test_get_data_new_no_new_data(self):
        """Verify get_data_new returns empty array when no new data.

        If called twice without adding data, second call returns empty array.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        # First call
        trigger.get_data_new()

        # Second call without new data
        empty = trigger.get_data_new()
        assert empty.shape == (0, 4)

    def test_get_data_new_before_trigger(self):
        """Verify get_data_new before trigger returns empty array.

        Before trigger fires, get_data_new should return (0, channels) array.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0)

        # Add data but don't trigger
        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is False
        empty = trigger.get_data_new()
        assert empty.shape == (0, 4)

    def test_get_data_new_plot_independent(self):
        """Verify N_new_samples_PLOT is independent of N_new_samples.

        Calling get_data_new should not affect N_new_samples_PLOT.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 trigger_type='up', presamples=10)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        n_new_plot_before = trigger.N_new_samples_PLOT

        # Get data for normal retrieval
        trigger.get_data_new()

        # N_new_samples_PLOT should be unchanged
        assert trigger.N_new_samples_PLOT == n_new_plot_before

    def test_get_data_new_plot_before_trigger(self):
        """Verify get_data_new_PLOT works before trigger fires.

        Unlike get_data_new, get_data_new_PLOT can return data before trigger.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0)

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.triggered is False
        plot_data = trigger.get_data_new_PLOT()

        # Should return the 100 samples added
        assert plot_data.shape[0] == 100
        assert plot_data.shape[1] == 4
        assert trigger.N_new_samples_PLOT == 0  # Reset after retrieval


# =============================================================================
# TestContinuousMode - Continuous Acquisition Mode
# =============================================================================

class TestContinuousMode:
    """Test continuous mode operation and buffer reset."""

    def test_continuous_mode_buffer_reset(self):
        """Verify continuous mode resets buffer when full.

        When buffer fills in continuous mode, reset_buffer() should be called
        and finished should remain False.
        """
        trigger = CustomPyTrigger(rows=100, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = None  # No limit

        # Trigger first
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)
        assert trigger.triggered is True

        # Add more data to exceed buffer capacity
        # After first add, buffer has ~35 samples used (rows_left ~65)
        # Adding 70 samples will exceed remaining space, triggering reset
        rows_left_before = trigger.rows_left
        data2 = make_ramp_data(10, 20, 70, 4)
        assert len(data2) > rows_left_before  # Ensure we exceed capacity
        trigger.add_data(data2)

        # Buffer should reset (verified by rows_left), not finish
        assert trigger.finished is False
        # After reset and adding data2, rows_left should be rows - len(data2)
        assert trigger.rows_left == trigger.rows - len(data2)

    def test_continuous_mode_duration_limit(self):
        """Verify continuous mode respects N_samples_to_acquire limit.

        When N_samples_to_acquire is set, finished should be True at limit.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = 200

        # Trigger
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)

        # Add data up to limit
        data2 = make_ramp_data(10, 20, 200, 4)
        trigger.add_data(data2)

        assert trigger.finished is True
        assert trigger.N_acquired_samples_since_trigger <= 200

    def test_continuous_mode_no_limit(self):
        """Verify continuous mode without limit continues indefinitely.

        With N_samples_to_acquire=None, can keep adding data past buffer size.
        """
        trigger = CustomPyTrigger(rows=100, channels=4, trigger_level=5.0,
                                 presamples=10)
        trigger.continuous_mode = True
        trigger.N_samples_to_acquire = None

        # Trigger
        data1 = make_ramp_data(0, 10, 50, 4)
        trigger.add_data(data1)

        # Add much more data than buffer size
        for _ in range(5):
            data = make_ramp_data(10, 20, 50, 4)
            trigger.add_data(data)

        assert trigger.finished is False


# =============================================================================
# TestGlobalTriggerSync - Global Trigger Synchronization
# =============================================================================

class TestGlobalTriggerSync:
    """Test global trigger synchronization across instances."""

    def test_first_instance_sets_global(self):
        """Verify first instance to trigger sets triggered_global.

        When instance A triggers, triggered_global should become True.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)

        assert CustomPyTrigger.triggered_global is False

        data = make_ramp_data(0, 10, 100, 4)
        trigger_a.add_data(data)

        assert trigger_a.triggered is True
        assert CustomPyTrigger.triggered_global is True

    def test_second_instance_receives_global(self):
        """Verify second instance receives global trigger.

        When instance A triggers, instance B should also trigger on next add_data.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)
        trigger_b = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                   presamples=10)  # Won't self-trigger

        # Trigger instance A
        data_a = make_ramp_data(0, 10, 100, 4)
        trigger_a.add_data(data_a)

        assert trigger_a.triggered is True
        assert CustomPyTrigger.triggered_global is True

        # Add data to instance B (won't cross its own threshold)
        data_b = make_ramp_data(0, 10, 100, 4)
        trigger_b.add_data(data_b)

        # Instance B should now be triggered via global
        assert trigger_b.triggered is True

    def test_n_triggers_both_instances(self):
        """Verify both instances increment N_triggers correctly.

        Both should have N_triggers=1 after global sync.
        """
        trigger_a = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                   presamples=10)
        trigger_b = CustomPyTrigger(rows=500, channels=4, trigger_level=100.0,
                                   presamples=10)

        # Trigger instance A
        data_a = make_ramp_data(0, 10, 100, 4)
        trigger_a.add_data(data_a)

        # Trigger instance B via global
        data_b = make_ramp_data(0, 10, 100, 4)
        trigger_b.add_data(data_b)

        assert trigger_a.N_triggers == 1
        assert trigger_b.N_triggers == 1


# =============================================================================
# TestTriggerCount - Trigger Counting Logic
# =============================================================================

class TestTriggerCount:
    """Test N_triggers counter and first_trigger flag."""

    def test_single_trigger_event(self):
        """Verify N_triggers increments once and first_trigger becomes False."""
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)

        assert trigger.first_trigger is True
        assert trigger.N_triggers == 0

        data = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data)

        assert trigger.N_triggers == 1
        assert trigger.first_trigger is False

    def test_no_double_count(self):
        """Verify N_triggers doesn't increment on subsequent data additions.

        After first trigger, N_triggers should stay at 1.
        """
        trigger = CustomPyTrigger(rows=500, channels=4, trigger_level=5.0,
                                 presamples=10)

        # First trigger
        data1 = make_ramp_data(0, 10, 100, 4)
        trigger.add_data(data1)
        assert trigger.N_triggers == 1

        # Add more data - should not increment N_triggers
        data2 = make_ramp_data(10, 20, 50, 4)
        trigger.add_data(data2)
        assert trigger.N_triggers == 1

        data3 = make_ramp_data(20, 30, 50, 4)
        trigger.add_data(data3)
        assert trigger.N_triggers == 1
