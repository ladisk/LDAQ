"""
Mock-based tests for ``LDAQ.national_instruments.NIAcquisition``.

These tests use ``unittest.mock.create_autospec(AITask)`` to enforce the
real ``nidaqwrapper.AITask`` signature, so any drift between
``NIAcquisition`` and the wrapper API surfaces as a test failure rather
than a silent hardware-only regression. They run without NI hardware and
are part of the standard ``pytest`` suite.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from nidaqwrapper import AITask

from LDAQ.national_instruments import NIAcquisition
from LDAQ.national_instruments import acquisition as ni_acq_mod


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_ai_task_mock(
    task_name: str = "test_task",
    sample_rate: float = 25_600.0,
    channel_list: list[str] | None = None,
) -> mock.MagicMock:
    """Build an autospec mock of ``AITask`` with the attributes ``NIAcquisition`` reads.

    ``create_autospec(AITask, instance=True)`` enforces real method
    signatures (so ``start(start_task=True)`` raises ``TypeError``), but
    plain attributes need to be set explicitly.
    """
    task = mock.create_autospec(AITask, instance=True)
    task.task_name = task_name
    task.sample_rate = sample_rate
    task.channel_list = list(channel_list) if channel_list is not None else ["ch1", "ch2"]
    # The underlying nidaqmx task object is accessed as ``self._ai_task.task``
    # for ``terminate_data_source()``. A plain MagicMock is fine here.
    task.task = mock.MagicMock(name="nidaqmx_task")
    return task


def _build_acquisition(task: mock.MagicMock, **kwargs) -> NIAcquisition:
    """Construct an ``NIAcquisition`` from an autospec'd ``AITask`` mock.

    ``NIAcquisition.__init__`` checks ``isinstance(task, AITask)``; the
    autospec mock is a real instance of ``AITask`` thanks to ``instance=True``.
    """
    return NIAcquisition(task, **kwargs)


@pytest.fixture
def ai_task() -> mock.MagicMock:
    return _make_ai_task_mock()


@pytest.fixture
def acq(ai_task: mock.MagicMock) -> NIAcquisition:
    return _build_acquisition(ai_task)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_aitask_instance_stored_unchanged(self):
        task = _make_ai_task_mock(task_name="my_task", sample_rate=25_600.0)
        acq = _build_acquisition(task)

        assert acq._ai_task is task
        assert acq.acquisition_name == "my_task"
        assert acq.sample_rate == 25_600.0

    def test_explicit_acquisition_name_overrides_task_name(self):
        task = _make_ai_task_mock(task_name="my_task")
        acq = _build_acquisition(task, acquisition_name="custom_name")

        assert acq.acquisition_name == "custom_name"

    def test_nimax_task_loaded_via_get_task_by_name(self):
        nidaqmx_task = mock.MagicMock(name="nidaqmx_task_from_max")
        wrapped_task = _make_ai_task_mock(task_name="VirtualTask")

        with (
            mock.patch.object(
                ni_acq_mod, "get_task_by_name", return_value=nidaqmx_task
            ) as get_task,
            mock.patch.object(
                ni_acq_mod.AITask, "from_task", return_value=wrapped_task
            ) as from_task,
        ):
            acq = NIAcquisition("VirtualTask")

        get_task.assert_called_once_with("VirtualTask")
        from_task.assert_called_once_with(nidaqmx_task)
        assert acq._ai_task is wrapped_task

    def test_unknown_nimax_task_raises_value_error(self):
        with mock.patch.object(ni_acq_mod, "get_task_by_name", return_value=None):
            with pytest.raises(ValueError, match="DoesNotExist"):
                NIAcquisition("DoesNotExist")

    def test_invalid_task_type_raises_type_error(self):
        with pytest.raises(TypeError, match="int"):
            NIAcquisition(42)


# ---------------------------------------------------------------------------
# Lifecycle: set_data_source / terminate_data_source
# ---------------------------------------------------------------------------


class TestSetDataSource:
    def test_first_call_starts_task_with_no_arguments(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        acq.set_data_source()

        ai_task.start.assert_called_once_with()
        assert acq._task_active is True

    def test_repeated_call_is_a_noop(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        acq.set_data_source()
        acq.set_data_source()

        ai_task.start.assert_called_once_with()

    def test_start_rejects_extra_kwargs_via_autospec(self, ai_task: mock.MagicMock):
        # Guards against the original bug: the previous implementation
        # called ``start(start_task=True)``, which the real ``AITask.start``
        # signature does not accept. ``create_autospec`` enforces this.
        with pytest.raises(TypeError):
            ai_task.start(start_task=True)


class TestTerminateDataSource:
    def test_stop_after_start_called_exactly_once(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        acq.set_data_source()
        acq.terminate_data_source()

        ai_task.task.stop.assert_called_once_with()
        assert acq._task_active is False

    def test_stop_without_start_is_noop(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        acq.terminate_data_source()

        ai_task.task.stop.assert_not_called()

    def test_repeated_stop_is_noop(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        acq.set_data_source()
        acq.terminate_data_source()
        acq.terminate_data_source()

        ai_task.task.stop.assert_called_once_with()


# ---------------------------------------------------------------------------
# read_data shape contract
# ---------------------------------------------------------------------------


class TestReadDataShape:
    def test_multichannel_shape_preserved_no_transpose(
        self, ai_task: mock.MagicMock
    ):
        # Distinct dimensions and a non-symmetric payload so a transpose
        # regression cannot pass by coincidence.
        payload = np.arange(200, dtype=float).reshape(100, 2)
        ai_task.acquire.return_value = payload

        acq = _build_acquisition(ai_task)
        result = acq.read_data()

        assert result.shape == (100, 2)
        np.testing.assert_array_equal(result, payload)

    def test_single_channel_shape_preserved(self):
        task = _make_ai_task_mock(channel_list=["ch1"])
        task.acquire.return_value = np.zeros((50, 1), dtype=float)

        acq = _build_acquisition(task)
        result = acq.read_data()

        assert result.shape == (50, 1)

    def test_empty_buffer_returns_zero_by_n_channels(
        self, ai_task: mock.MagicMock
    ):
        ai_task.acquire.return_value = np.empty((0, 2), dtype=float)

        acq = _build_acquisition(ai_task)
        result = acq.read_data()

        assert result.shape == (0, 2)
        assert result.ndim == 2

    def test_none_return_treated_as_empty(self, ai_task: mock.MagicMock):
        ai_task.acquire.return_value = None

        acq = _build_acquisition(ai_task)
        result = acq.read_data()

        assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# clear_buffer
# ---------------------------------------------------------------------------


class TestClearBuffer:
    def test_calls_acquire_with_n_samples_none(
        self, acq: NIAcquisition, ai_task: mock.MagicMock
    ):
        ai_task.acquire.reset_mock()  # ignore any constructor-time calls
        acq.clear_buffer()

        ai_task.acquire.assert_called_once_with(n_samples=None)


# ---------------------------------------------------------------------------
# ImportError when nidaqwrapper is missing
# ---------------------------------------------------------------------------


class TestMissingNidaqwrapper:
    def test_constructor_raises_import_error(self, monkeypatch):
        monkeypatch.setattr(ni_acq_mod, "_NIDAQWRAPPER_AVAILABLE", False)

        with pytest.raises(ImportError, match="nidaqwrapper"):
            NIAcquisition(_make_ai_task_mock())
