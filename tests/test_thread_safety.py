"""
Thread Safety Tests for LDAQ

Tests verify thread-safe behavior for concurrent acquisition operations.
These tests specify CORRECT behavior - some may fail until fixes are applied.

Notes
-----
This is TDD (Test-Driven Development). These tests specify the CONTRACT for
thread-safe operation. The current implementation has known race conditions
that may cause some tests to fail. That's expected - the tests define what
should be fixed.

Key Testing Principles
----------------------
1. Tests calculate expected outcomes independently (not from code under test)
2. Use threading primitives (Event, Barrier) to avoid flaky race detection
3. Run stress tests multiple iterations to increase race condition detection
4. All tests use SimulatedAcquisition (no hardware required)
"""

from __future__ import annotations

import threading
import time
from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import LDAQ
from LDAQ.acquisition_base import BaseAcquisition, CustomPyTrigger


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def create_acquisition():
    """
    Factory to create configured SimulatedAcquisition instances.

    Returns
    -------
    Callable[[str, float, float], LDAQ.simulator.SimulatedAcquisition]
        Factory function that creates a SimulatedAcquisition with:
        - name: acquisition name
        - sample_rate: sampling rate in Hz
        - duration_data: duration of simulated data in seconds
    """

    def _create(
        name: str = "test_acq", sample_rate: float = 1000, duration_data: float = 10.0
    ) -> LDAQ.simulator.SimulatedAcquisition:
        acq = LDAQ.simulator.SimulatedAcquisition(acquisition_name=name)
        n_samples = int(duration_data * sample_rate)
        t = np.arange(n_samples) / sample_rate
        # 10 Hz sine wave
        data = np.sin(2 * np.pi * 10 * t).reshape(-1, 1)
        acq.set_simulated_data(
            data, channel_names=[f"{name}_ch0"], sample_rate=sample_rate
        )
        return acq

    return _create


@pytest.fixture
def reset_global_state():
    """
    Reset global class variables before and after tests.

    Yields
    ------
    None
        Ensures clean state for each test.
    """
    # Reset before test
    CustomPyTrigger.clear_triggered_global()
    BaseAcquisition.reset_ready_state(expected_count=0)

    yield

    # Reset after test
    CustomPyTrigger.clear_triggered_global()
    BaseAcquisition.reset_ready_state(expected_count=0)


# =============================================================================
# Test: Global Trigger Coordination
# =============================================================================


class TestGlobalTriggerCoordination:
    """
    Tests for thread-safe triggered_global coordination.

    The triggered_global class variable is shared across all acquisition
    instances and must be safely readable/writable from multiple threads.
    """

    def test_concurrent_trigger_all_observe(
        self, create_acquisition, reset_global_state
    ):
        """
        When one source triggers globally, all other sources observe it.

        Contract
        --------
        - Multiple acquisition sources running concurrently
        - When triggered_global is set to True, all sources must observe it
        - No source should miss the global trigger event
        - Expected: All sources observe triggered_global = True within 100ms

        Notes
        -----
        This test may FAIL on current implementation due to race conditions.
        """
        n_sources = 3
        sources = [create_acquisition(f"src{i}", sample_rate=100) for i in range(n_sources)]

        # Configure all sources with trigger that won't fire naturally
        for src in sources:
            src.set_trigger(level=1e20, channel=0, duration=0.5, presamples=10)
            src.set_data_source()

        # Synchronization primitives
        start_barrier = threading.Barrier(n_sources + 1)  # +1 for main thread
        observed_trigger = [False] * n_sources
        errors = []

        def run_source(src: BaseAcquisition, index: int) -> None:
            """Run acquisition and check if global trigger is observed."""
            try:
                start_barrier.wait()  # Sync start
                # Run briefly and check for global trigger
                start_time = time.time()
                while time.time() - start_time < 0.2:
                    if CustomPyTrigger.triggered_global:
                        observed_trigger[index] = True
                        break
                    time.sleep(0.001)  # Brief sleep to yield
            except Exception as e:
                errors.append((index, e))
            finally:
                try:
                    src.stop()
                    src.terminate_data_source()
                except Exception:
                    pass

        # Start all threads
        threads = [
            threading.Thread(target=run_source, args=(s, i), daemon=True)
            for i, s in enumerate(sources)
        ]
        for t in threads:
            t.start()

        # Wait for all threads to reach barrier, then trigger globally
        start_barrier.wait()
        time.sleep(0.05)  # Let threads start their loops

        # Set global trigger
        CustomPyTrigger.triggered_global = True

        # Wait for threads to complete
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), "Thread did not terminate within timeout"

        # Verify results
        assert not errors, f"Errors occurred in threads: {errors}"
        # CONTRACT: All sources must observe the global trigger
        assert all(
            observed_trigger
        ), f"Not all sources observed global trigger: {observed_trigger}"

    def test_trigger_state_consistency_no_torn_reads(
        self, create_acquisition, reset_global_state
    ):
        """
        Reading triggered_global returns consistent boolean (no torn reads).

        Contract
        --------
        - Reading triggered_global from multiple threads concurrently
        - Must always return a valid boolean (True or False)
        - Never return inconsistent/corrupted value
        - Expected: 1000 concurrent reads all return valid bool

        Notes
        -----
        Python bool assignment is atomic, but this verifies no GIL issues.
        """
        n_readers = 10
        n_reads_per_reader = 100
        read_values = [[] for _ in range(n_readers)]
        errors = []

        start_barrier = threading.Barrier(n_readers + 1)

        def read_trigger_state(reader_index: int) -> None:
            """Repeatedly read triggered_global and record values."""
            try:
                start_barrier.wait()
                for _ in range(n_reads_per_reader):
                    value = CustomPyTrigger.triggered_global
                    # Verify it's a valid boolean
                    assert isinstance(
                        value, bool
                    ), f"triggered_global is not bool: {type(value)}"
                    read_values[reader_index].append(value)
            except Exception as e:
                errors.append((reader_index, e))

        # Start reader threads
        threads = [
            threading.Thread(target=read_trigger_state, args=(i,), daemon=True)
            for i in range(n_readers)
        ]
        for t in threads:
            t.start()

        # Start all readers, then toggle trigger state
        start_barrier.wait()
        for _ in range(5):
            time.sleep(0.001)
            CustomPyTrigger.triggered_global = not CustomPyTrigger.triggered_global

        for t in threads:
            t.join(timeout=5)

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: Every read must be a valid boolean
        for reader_index, values in enumerate(read_values):
            assert len(values) == n_reads_per_reader, f"Reader {reader_index} missing reads"
            assert all(
                isinstance(v, bool) for v in values
            ), f"Reader {reader_index} got non-bool values"

    def test_trigger_propagation_race_free(
        self, create_acquisition, reset_global_state
    ):
        """
        Multiple sources triggering simultaneously - last write wins cleanly.

        Contract
        --------
        - Multiple threads attempting to set triggered_global = True
        - All writes succeed without corruption
        - Final state is True (at least one write succeeded)
        - Expected: triggered_global = True after concurrent writes

        Notes
        -----
        Verifies write operations don't corrupt the shared variable.
        """
        n_writers = 5
        errors = []

        start_barrier = threading.Barrier(n_writers + 1)

        def trigger_globally(writer_index: int) -> None:
            """Attempt to set global trigger."""
            try:
                start_barrier.wait()
                CustomPyTrigger.triggered_global = True
            except Exception as e:
                errors.append((writer_index, e))

        threads = [
            threading.Thread(target=trigger_globally, args=(i,), daemon=True)
            for i in range(n_writers)
        ]
        for t in threads:
            t.start()

        start_barrier.wait()  # Start all writers simultaneously

        for t in threads:
            t.join(timeout=5)

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: Final state must be True (at least one write succeeded)
        assert (
            CustomPyTrigger.triggered_global is True
        ), "triggered_global not set after concurrent writes"


# =============================================================================
# Test: All Acquisitions Ready Coordination
# =============================================================================


class TestAllAcquisitionsReady:
    """
    Tests for thread-safe all_acquisitions_ready flag.

    The all_acquisitions_ready class variable is a boolean flag used by Core
    to signal when all acquisition sources are ready to start.

    Note: The actual usage in LDAQ is boolean (True/False), not a counter.
    Core checks if all sources have is_ready=True, then sets all_acquisitions_ready=True.
    """

    def test_concurrent_ready_signal_via_signal_ready(self, reset_global_state):
        """
        Multiple sources using signal_ready() - event is set when all ready.

        Contract
        --------
        - N sources call signal_ready() concurrently
        - When expected count is reached, are_all_ready() returns True
        - Expected: are_all_ready() == True after all sources signal
        """
        n_sources = 10
        errors = []

        # Initialize expected count
        BaseAcquisition.reset_ready_state(expected_count=n_sources)

        start_barrier = threading.Barrier(n_sources + 1)

        def signal_ready_fn(index: int) -> None:
            """Signal this source is ready using thread-safe method."""
            try:
                start_barrier.wait()
                BaseAcquisition.signal_ready()
            except Exception as e:
                errors.append((index, e))

        threads = [
            threading.Thread(target=signal_ready_fn, args=(i,), daemon=True)
            for i in range(n_sources)
        ]
        for t in threads:
            t.start()

        start_barrier.wait()  # Start all signals simultaneously

        for t in threads:
            t.join(timeout=5)

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: All sources signaled, so are_all_ready() must be True
        assert BaseAcquisition.are_all_ready(), (
            f"are_all_ready() should be True after {n_sources} sources signaled"
        )

    def test_ready_flag_stress_test(self, reset_global_state):
        """
        Stress test: 50 iterations of concurrent boolean flag setting.

        Contract
        --------
        - Run 50 iterations of concurrent True setting
        - Final state should always be True after concurrent sets
        - Expected: No exceptions or corrupted state
        """
        n_writers = 5
        n_iterations = 50
        errors = []

        for iteration in range(n_iterations):
            BaseAcquisition.all_acquisitions_ready = False
            start_barrier = threading.Barrier(n_writers + 1)

            def set_ready() -> None:
                try:
                    start_barrier.wait()
                    # Simulate Core setting all ready
                    BaseAcquisition.all_acquisitions_ready = True
                except Exception as e:
                    errors.append((iteration, e))

            threads = [
                threading.Thread(target=set_ready, daemon=True)
                for _ in range(n_writers)
            ]
            for t in threads:
                t.start()

            start_barrier.wait()

            for t in threads:
                t.join(timeout=5)

            # After all threads set True, it must be True
            if not BaseAcquisition.all_acquisitions_ready:
                errors.append((iteration, "Flag not True after concurrent sets"))

        # CONTRACT: No errors in any iteration
        assert len(errors) == 0, f"Errors in {len(errors)} iterations: {errors[:5]}"


# =============================================================================
# Test: Lock Protection
# =============================================================================


class TestLockProtection:
    """
    Tests for lock_acquisition protecting shared state.

    The lock_acquisition lock must protect concurrent access to:
    - Ring buffer data (Rone, Rtwo)
    - Virtual channel definitions
    - Acquisition state
    """

    def test_concurrent_data_read_during_acquisition(
        self, create_acquisition, reset_global_state
    ):
        """
        Concurrent get_data() during acquisition returns consistent snapshots.

        Contract
        --------
        - Acquisition thread writing data continuously
        - Multiple reader threads calling get_data() concurrently
        - Each reader must get consistent data (no torn reads)
        - Expected: All data arrays have correct shape and valid measurement dict

        Notes
        -----
        Tests that lock_acquisition is used consistently in read/write paths.
        """
        acq = create_acquisition("concurrent_read_test", sample_rate=1000)
        acq.set_trigger(level=0.5, channel=0, duration=0.2, presamples=50)
        acq.set_data_source()

        n_readers = 5
        n_reads_per_reader = 20
        errors = []
        successful_reads = [0] * n_readers

        start_event = threading.Event()
        stop_event = threading.Event()

        def acquisition_thread() -> None:
            """Run acquisition in background."""
            try:
                start_event.set()
                acq.run_acquisition(run_time=2.0)
            except Exception as e:
                errors.append(("acquisition", e))
            finally:
                stop_event.set()

        def reader_thread(reader_index: int) -> None:
            """Repeatedly read data concurrently."""
            try:
                start_event.wait()  # Wait for acquisition to start
                time.sleep(0.05)  # Let some data accumulate

                for _ in range(n_reads_per_reader):
                    if stop_event.is_set():
                        break

                    result = acq.get_measurement_dict()

                    # CONTRACT: Must return valid measurement dict structure
                    assert "time" in result, "Missing 'time' key"
                    assert "data" in result, "Missing 'data' key"
                    assert "channel_names" in result, "Missing 'channel_names' key"

                    time_array = result["time"]
                    data_array = result["data"]

                    # CONTRACT: Valid array shapes
                    assert time_array.ndim == 1, f"time not 1D: {time_array.shape}"
                    assert data_array.ndim == 2, f"data not 2D: {data_array.shape}"
                    assert (
                        data_array.shape[0] == time_array.shape[0]
                    ), "data/time length mismatch"

                    successful_reads[reader_index] += 1
                    time.sleep(0.01)  # Brief pause between reads

            except Exception as e:
                errors.append((reader_index, e))

        # Start acquisition thread
        acq_thread = threading.Thread(target=acquisition_thread, daemon=True)
        acq_thread.start()

        # Start reader threads
        reader_threads = [
            threading.Thread(target=reader_thread, args=(i,), daemon=True)
            for i in range(n_readers)
        ]
        for t in reader_threads:
            t.start()

        # Wait for completion
        acq_thread.join(timeout=10)
        for t in reader_threads:
            t.join(timeout=5)

        # Cleanup
        acq.stop()
        acq.terminate_data_source()

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: All readers should complete some successful reads
        assert all(
            count > 0 for count in successful_reads
        ), f"Some readers failed to read data: {successful_reads}"

    def test_add_virtual_channel_during_acquisition_is_atomic(
        self, create_acquisition, reset_global_state
    ):
        """
        Adding virtual channel during acquisition is atomic (lock protected).

        Contract
        --------
        - Acquisition running in background
        - Add virtual channel from main thread
        - Virtual channel should be available without corrupting acquisition
        - Expected: Virtual channel appears in channel_names after addition

        Notes
        -----
        Tests that add_virtual_channel uses lock_acquisition properly.
        """
        acq = create_acquisition("virtual_channel_test", sample_rate=1000)
        acq.set_trigger(level=0.5, channel=0, duration=0.5, presamples=50)
        acq.set_data_source()

        errors = []
        virtual_channel_added = threading.Event()

        def acquisition_thread() -> None:
            """Run acquisition in background."""
            try:
                acq.run_acquisition(run_time=1.0)
            except Exception as e:
                errors.append(("acquisition", e))

        # Start acquisition
        acq_thread = threading.Thread(target=acquisition_thread, daemon=True)
        acq_thread.start()

        time.sleep(0.1)  # Let acquisition start

        # Add virtual channel during acquisition
        def square_channel(ch0):
            """Square the first channel."""
            return ch0 ** 2

        acq.add_virtual_channel("squared", "virtual_channel_test_ch0", square_channel)
        virtual_channel_added.set()

        # Wait for acquisition to complete
        acq_thread.join(timeout=5)

        # Cleanup
        acq.stop()
        acq.terminate_data_source()

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: Virtual channel must appear in channel_names
        assert (
            "squared" in acq.channel_names
        ), "Virtual channel not added to channel_names"

    def test_lock_acquisition_actually_locks(
        self, create_acquisition, reset_global_state
    ):
        """
        Verify lock_acquisition actually provides mutual exclusion.

        Contract
        --------
        - Multiple threads attempting to acquire lock_acquisition
        - Only one thread holds lock at a time
        - Expected: Critical section executes atomically

        Notes
        -----
        Tests that lock_acquisition behaves as a proper lock.
        """
        acq = create_acquisition("lock_test")

        n_threads = 5
        critical_section_count = [0]  # Wrapped in list to avoid closure issues
        max_concurrent = [0]

        start_barrier = threading.Barrier(n_threads + 1)
        errors = []

        def worker(index: int) -> None:
            """Acquire lock and update shared counters."""
            try:
                start_barrier.wait()

                with acq.lock_acquisition:
                    # Enter critical section
                    critical_section_count[0] += 1
                    current_count = critical_section_count[0]
                    max_concurrent[0] = max(max_concurrent[0], current_count)

                    # Simulate some work
                    time.sleep(0.01)

                    # Exit critical section
                    critical_section_count[0] -= 1

            except Exception as e:
                errors.append((index, e))

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()

        start_barrier.wait()

        for t in threads:
            t.join(timeout=5)

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: Lock must ensure only one thread in critical section
        assert (
            max_concurrent[0] == 1
        ), f"Lock failed: {max_concurrent[0]} threads in critical section"


# =============================================================================
# Test: Thread Timeout
# =============================================================================


class TestThreadTimeout:
    """
    Tests for acquisition thread termination with timeout.

    Acquisition threads must terminate within a reasonable timeout when
    stop_acquisition() is called. Hanging threads indicate missing checks
    of _running flag or blocked I/O.
    """

    def test_stop_acquisition_terminates_within_timeout(
        self, create_acquisition, reset_global_state
    ):
        """
        stop_acquisition() causes thread to terminate within 5 seconds.

        Contract
        --------
        - Start acquisition in thread
        - Call stop_acquisition()
        - Thread must terminate within 5 seconds
        - Expected: Thread.join(timeout=5) succeeds

        Notes
        -----
        Current implementation may not meet this contract if _running flag
        is not checked frequently enough in acquisition loop.
        """
        acq = create_acquisition("timeout_test", sample_rate=1000)
        # No trigger - continuous acquisition
        acq.set_trigger(level=1e20, channel=0, duration=10.0, presamples=10)
        acq.set_data_source()

        def acquisition_thread() -> None:
            """Run long acquisition."""
            acq.run_acquisition(run_time=100.0)  # Very long duration

        thread = threading.Thread(target=acquisition_thread, daemon=True)
        thread.start()

        time.sleep(0.2)  # Let acquisition start

        # Stop acquisition
        stop_time = time.time()
        acq.stop()

        # CONTRACT: Thread must terminate within 5 seconds
        thread.join(timeout=5)
        elapsed = time.time() - stop_time

        # Cleanup
        try:
            acq.terminate_data_source()
        except Exception:
            pass

        # Verify termination
        assert not thread.is_alive(), (
            f"Thread did not terminate within 5s timeout (elapsed: {elapsed:.2f}s)"
        )

    def test_thread_join_without_timeout_risk(
        self, create_acquisition, reset_global_state
    ):
        """
        Verify acquisition can be stopped cleanly without indefinite hang.

        Contract
        --------
        - Start acquisition
        - Stop acquisition with stop_acquisition()
        - Join with timeout must succeed (not hang indefinitely)
        - Expected: Clean shutdown within timeout

        Notes
        -----
        Tests the actual pattern used in Core.run() where thread.join() is
        called. Without timeout, this could hang forever.
        """
        acq = create_acquisition("join_test", sample_rate=500)
        acq.set_trigger(level=1e20, channel=0, duration=5.0, presamples=10)
        acq.set_data_source()

        stopped_cleanly = threading.Event()

        def acquisition_thread() -> None:
            """Run acquisition and signal when stopped."""
            try:
                acq.run_acquisition(run_time=100.0)
            finally:
                stopped_cleanly.set()

        thread = threading.Thread(target=acquisition_thread, daemon=True)
        thread.start()

        time.sleep(0.1)  # Let acquisition start

        # Stop and join with timeout
        acq.stop()
        thread.join(timeout=3)

        # Cleanup
        try:
            acq.terminate_data_source()
        except Exception:
            pass

        # CONTRACT: Thread must have stopped and signaled
        assert stopped_cleanly.is_set(), "Acquisition did not stop cleanly"
        assert not thread.is_alive(), "Thread still alive after join timeout"


# =============================================================================
# Test: Resource Cleanup
# =============================================================================


class TestResourceCleanup:
    """
    Tests for guaranteed resource cleanup on errors and stop.

    terminate_data_source() must be called to release hardware resources,
    even when exceptions occur during acquisition.
    """

    def test_cleanup_on_read_data_exception(
        self, create_acquisition, reset_global_state
    ):
        """
        terminate_data_source called when read_data() raises exception.

        Contract
        --------
        - Acquisition running
        - read_data() raises exception
        - terminate_data_source() must still be called
        - Expected: terminate_data_source called despite exception

        Notes
        -----
        Current implementation may not guarantee this in acquire() method.
        This test is EXPECTED TO FAIL until Phase 2 fixes are applied.
        """
        acq = create_acquisition("cleanup_exception_test", sample_rate=1000)
        acq.set_trigger(level=0.0, channel=0, duration=0.1, presamples=10)
        acq.set_data_source()

        cleanup_called = threading.Event()
        original_terminate = acq.terminate_data_source

        def mock_terminate() -> None:
            """Track if terminate is called."""
            cleanup_called.set()
            original_terminate()

        acq.terminate_data_source = mock_terminate

        # Patch read_data to raise exception
        original_read_data = acq.read_data

        def failing_read_data() -> np.ndarray:
            """Raise exception on read."""
            raise RuntimeError("Simulated hardware error")

        acq.read_data = failing_read_data

        # Run acquisition - should raise but still cleanup
        exception_raised = False
        try:
            acq.run_acquisition(run_time=0.2)
        except RuntimeError:
            exception_raised = True

        # Wait a moment for cleanup
        time.sleep(0.1)

        # Restore original method
        acq.read_data = original_read_data

        # CONTRACT: terminate_data_source must be called despite exception
        assert exception_raised, "Expected RuntimeError was not raised"
        assert cleanup_called.is_set(), (
            "terminate_data_source was not called after read_data exception"
        )

    def test_cleanup_on_stop_during_active_acquisition(
        self, create_acquisition, reset_global_state
    ):
        """
        stop_acquisition during active run calls terminate_data_source.

        Contract
        --------
        - Acquisition running normally
        - stop_acquisition() called from another thread
        - terminate_data_source() must be called after stop
        - Expected: Clean shutdown with resource cleanup

        Notes
        -----
        Tests normal shutdown path with cleanup guarantee.
        """
        acq = create_acquisition("cleanup_stop_test", sample_rate=1000)
        acq.set_trigger(level=1e20, channel=0, duration=1.0, presamples=10)
        acq.set_data_source()

        cleanup_called = threading.Event()
        original_terminate = acq.terminate_data_source

        def mock_terminate() -> None:
            """Track if terminate is called."""
            cleanup_called.set()
            original_terminate()

        acq.terminate_data_source = mock_terminate

        def acquisition_thread() -> None:
            """Run acquisition."""
            acq.run_acquisition(run_time=10.0)

        thread = threading.Thread(target=acquisition_thread, daemon=True)
        thread.start()

        time.sleep(0.2)  # Let acquisition start

        # Stop from main thread
        acq.stop()

        # Wait for thread to finish
        thread.join(timeout=5)

        # CONTRACT: terminate_data_source called after stop
        assert cleanup_called.is_set(), (
            "terminate_data_source not called after stop_acquisition"
        )

    def test_cleanup_guarantee_with_context_manager_pattern(
        self, create_acquisition, reset_global_state
    ):
        """
        Verify cleanup works similar to context manager (try/finally).

        Contract
        --------
        - Even with exception, cleanup must execute
        - Pattern should be: try: acquire() finally: terminate()
        - Expected: Cleanup called in all cases

        Notes
        -----
        Tests that acquisition methods follow try/finally pattern internally.
        This test is EXPECTED TO FAIL until Phase 2 fixes are applied.
        """
        acq = create_acquisition("cleanup_guarantee_test", sample_rate=500)

        cleanup_count = [0]

        def counting_terminate() -> None:
            """Count cleanup calls."""
            cleanup_count[0] += 1

        acq.terminate_data_source = counting_terminate

        # Test 1: Normal execution
        acq.set_trigger(level=1e20, channel=0, duration=0.1, presamples=10)
        acq.set_data_source()
        acq.run_acquisition(run_time=0.1)
        time.sleep(0.2)

        normal_count = cleanup_count[0]

        # Test 2: With exception
        def failing_read() -> np.ndarray:
            raise ValueError("Test exception")

        acq.read_data = failing_read
        acq.set_data_source()

        try:
            acq.run_acquisition(run_time=0.1)
        except (ValueError, RuntimeError):
            pass  # Expected

        time.sleep(0.1)

        exception_count = cleanup_count[0]

        # CONTRACT: Cleanup must be called in both normal and exception cases
        assert normal_count >= 1, "Cleanup not called in normal case"
        assert exception_count > normal_count, (
            "Cleanup not called after exception"
        )


# =============================================================================
# Test: Simulator Thread Safety
# =============================================================================


class TestSimulatorThreadSafety:
    """
    Tests for thread-safe buffer access in SimulatedAcquisition.

    The simulator uses a child process to generate data. Access to
    shared buffers and flags must be thread-safe.
    """

    def test_read_data_during_buffer_update(
        self, create_acquisition, reset_global_state
    ):
        """
        read_data() during buffer update returns consistent data.

        Contract
        --------
        - Simulator child process updating buffer continuously
        - Multiple threads calling read_data() concurrently
        - Each read must return valid data (no partial/corrupted reads)
        - Expected: All reads return correct shape and finite values

        Notes
        -----
        Tests that simulator's buffer access is protected.
        """
        acq = create_acquisition("sim_buffer_test", sample_rate=1000, duration_data=5.0)
        acq.set_trigger(level=1e20, channel=0, duration=0.5, presamples=10)
        acq.set_data_source()

        n_readers = 3
        n_reads_per_reader = 30
        errors = []
        valid_reads = [0] * n_readers

        start_event = threading.Event()
        stop_event = threading.Event()

        def reader_thread(reader_index: int) -> None:
            """Repeatedly call read_data()."""
            try:
                start_event.wait()

                for _ in range(n_reads_per_reader):
                    if stop_event.is_set():
                        break

                    data = acq.read_data()

                    # CONTRACT: Valid data shape and values
                    assert isinstance(data, np.ndarray), f"Not ndarray: {type(data)}"
                    assert data.ndim == 2, f"Not 2D: {data.shape}"
                    assert data.shape[1] == 1, f"Wrong channel count: {data.shape}"
                    assert np.all(np.isfinite(data)), "Non-finite values in data"

                    valid_reads[reader_index] += 1
                    time.sleep(0.01)

            except Exception as e:
                errors.append((reader_index, e))

        # Start readers
        threads = [
            threading.Thread(target=reader_thread, args=(i,), daemon=True)
            for i in range(n_readers)
        ]
        for t in threads:
            t.start()

        start_event.set()
        time.sleep(1.0)  # Let readers run
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        # Cleanup
        acq.stop()
        acq.terminate_data_source()

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: All readers should complete reads successfully
        assert all(count > 0 for count in valid_reads), (
            f"Some readers failed: {valid_reads}"
        )

    def test_child_process_started_flag_atomic_access(
        self, create_acquisition, reset_global_state
    ):
        """
        child_process_started flag access is atomic.

        Contract
        --------
        - Multiple threads checking child_process_started flag
        - Flag must always return consistent boolean value
        - Expected: All reads return valid bool

        Notes
        -----
        Similar to triggered_global test but for simulator-specific flag.
        """
        acq = create_acquisition("sim_flag_test")

        n_readers = 10
        n_reads_per_reader = 50
        read_values = [[] for _ in range(n_readers)]
        errors = []

        # Set data source to initialize child process
        acq.set_data_source()

        start_barrier = threading.Barrier(n_readers + 1)

        def read_flag(reader_index: int) -> None:
            """Repeatedly read child_process_started flag."""
            try:
                start_barrier.wait()
                for _ in range(n_reads_per_reader):
                    value = acq.child_process_started
                    assert isinstance(value, bool), f"Not bool: {type(value)}"
                    read_values[reader_index].append(value)
            except Exception as e:
                errors.append((reader_index, e))

        threads = [
            threading.Thread(target=read_flag, args=(i,), daemon=True)
            for i in range(n_readers)
        ]
        for t in threads:
            t.start()

        start_barrier.wait()

        for t in threads:
            t.join(timeout=5)

        # Cleanup
        acq.terminate_data_source()

        # Verify results
        assert not errors, f"Errors occurred: {errors}"
        # CONTRACT: All reads must be valid booleans
        for reader_index, values in enumerate(read_values):
            assert len(values) == n_reads_per_reader, (
                f"Reader {reader_index} missing reads"
            )
            assert all(isinstance(v, bool) for v in values), (
                f"Reader {reader_index} got non-bool values"
            )


# =============================================================================
# Test: Stress/Concurrency
# =============================================================================


class TestStressConcurrency:
    """
    Stress tests with many iterations to detect intermittent race conditions.

    These tests run scenarios many times to increase probability of
    detecting timing-dependent bugs.
    """

    def test_stress_multi_source_concurrent_acquisition(self, reset_global_state):
        """
        Stress test: Multiple sources acquiring concurrently (20 iterations).

        Contract
        --------
        - 3 sources acquiring simultaneously for 20 iterations
        - All sources must complete without errors or exceptions
        - Expected: 100% success rate across all iterations

        Notes
        -----
        This is a comprehensive stress test that may reveal race conditions
        in trigger coordination, ready signaling, or data access.
        """
        n_sources = 3
        n_iterations = 20
        failures = []

        for iteration in range(n_iterations):
            # Reset global state each iteration
            CustomPyTrigger.triggered_global = False
            BaseAcquisition.all_acquisitions_ready = 0

            sources = []
            for i in range(n_sources):
                acq = LDAQ.simulator.SimulatedAcquisition(
                    acquisition_name=f"stress_{iteration}_{i}"
                )
                t = np.arange(1000) / 1000.0
                data = np.sin(2 * np.pi * 10 * t).reshape(-1, 1)
                acq.set_simulated_data(
                    data, channel_names=[f"ch{i}"], sample_rate=1000
                )
                acq.set_trigger(level=0.5, channel=0, duration=0.2, presamples=20)
                sources.append(acq)

            errors = []

            def run_source(src: BaseAcquisition, index: int) -> None:
                """Run acquisition concurrently."""
                try:
                    src.set_data_source()
                    src.run_acquisition(run_time=1.0)
                except Exception as e:
                    errors.append((iteration, index, e))
                finally:
                    try:
                        src.stop()
                        src.terminate_data_source()
                    except Exception:
                        pass

            threads = [
                threading.Thread(target=run_source, args=(s, i), daemon=True)
                for i, s in enumerate(sources)
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=10)

            # Check for failures
            if errors:
                failures.append((iteration, "errors", errors))

        # CONTRACT: All iterations must succeed
        if failures:
            summary = f"{len(failures)}/{n_iterations} iterations failed\n"
            summary += f"First 5 failures: {failures[:5]}"
            pytest.fail(summary)

    def test_stress_rapid_start_stop_cycles(
        self, create_acquisition, reset_global_state
    ):
        """
        Stress test: Rapid start/stop cycles (30 iterations).

        Contract
        --------
        - Start and immediately stop acquisition 30 times
        - All cycles must complete without errors
        - No resource leaks or hanging threads
        - Expected: 100% success rate

        Notes
        -----
        Tests robustness of acquisition lifecycle and cleanup.
        May reveal issues with thread coordination or resource cleanup.
        """
        n_iterations = 30
        failures = []

        acq = create_acquisition("rapid_cycle_test", sample_rate=500)

        for iteration in range(n_iterations):
            try:
                acq.set_trigger(level=1e20, channel=0, duration=0.1, presamples=10)
                acq.set_data_source()

                def run_briefly() -> None:
                    """Run acquisition briefly."""
                    acq.run_acquisition(run_time=10.0)  # Long, will be stopped

                thread = threading.Thread(target=run_briefly, daemon=True)
                thread.start()

                time.sleep(0.01)  # Let it start

                # Immediately stop
                acq.stop()

                # Must terminate quickly
                thread.join(timeout=2)

                if thread.is_alive():
                    failures.append((iteration, "thread_did_not_stop"))
                    continue

                acq.terminate_data_source()

            except Exception as e:
                failures.append((iteration, e))

        # CONTRACT: All cycles must complete successfully
        if failures:
            summary = f"{len(failures)}/{n_iterations} cycles failed\n"
            summary += f"First 5 failures: {failures[:5]}"
            pytest.fail(summary)
