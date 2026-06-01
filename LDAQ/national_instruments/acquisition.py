from __future__ import annotations

import numpy as np

from ..acquisition_base import BaseAcquisition

_NIDAQWRAPPER_AVAILABLE = False
try:
    from nidaqwrapper import AITask, get_task_by_name
    _NIDAQWRAPPER_AVAILABLE = True
except ImportError:
    pass


class NIAcquisition(BaseAcquisition):
    """Acquisition class for National Instruments devices via nidaqwrapper.

    A thin wrapper around ``nidaqwrapper.AITask`` that satisfies the
    ``BaseAcquisition`` contract. Supports both programmatic tasks
    (``AITask`` objects) and tasks defined in NI MAX (task name strings).

    Parameters
    ----------
    task : nidaqwrapper.AITask or str
        Either a fully configured ``AITask`` instance or the name of a task
        saved in NI MAX.  When a string is supplied the task is loaded via
        ``nidaqwrapper.get_task_by_name()``.
    acquisition_name : str or None, optional
        Human-readable name for this acquisition source.  Defaults to the
        underlying NI task name when ``None``.

    Raises
    ------
    ImportError
        If the ``nidaqwrapper`` package is not installed.
    TypeError
        If ``task`` is not an ``AITask`` instance or a string.

    Examples
    --------
    Wrap a programmatically created task:

    >>> ai = AITask("my_task", sample_rate=10_000)
    >>> ai.add_channel(...)
    >>> acq = NIAcquisition(ai)

    Load a task saved in NI MAX:

    >>> acq = NIAcquisition("MyNIMaxTask")
    """

    def __init__(
        self,
        task: AITask | str,
        acquisition_name: str | None = None,
    ) -> None:
        """
        Initialize NIAcquisition.

        Parameters
        ----------
        task : nidaqwrapper.AITask or str
            Source task: either an ``AITask`` object or an NI MAX task name.
        acquisition_name : str or None, optional
            Name for this acquisition source.  Uses the task name when None.

        Raises
        ------
        ImportError
            If ``nidaqwrapper`` is not installed.
        TypeError
            If ``task`` is neither an ``AITask`` nor a string.
        """
        if not _NIDAQWRAPPER_AVAILABLE:
            raise ImportError(
                "nidaqwrapper is not installed. "
                "Install it before using NIAcquisition."
            )

        super().__init__()

        if isinstance(task, AITask):
            self._ai_task: AITask = task
        elif isinstance(task, str):
            ni_task = get_task_by_name(task)
            if ni_task is None:
                raise ValueError(f"NI MAX task '{task}' not found.")
            self._ai_task = AITask.from_task(ni_task)
        else:
            raise TypeError(
                f"task must be an AITask instance or a string (NI MAX task name), "
                f"got {type(task).__name__!r}."
            )

        self.acquisition_name = (
            acquisition_name if acquisition_name is not None else self._ai_task.task_name
        )
        self.sample_rate = self._ai_task.sample_rate
        self._channel_names_init = list(self._ai_task.channel_list)
        self._task_active = False

        self._set_all_channels()  # populate channel_names_all before set_trigger
        self.set_trigger(1e20, 0, duration=1.0)

    def set_data_source(self) -> None:
        """Start the underlying AITask in preparation for acquisition.

        Calls ``super().set_data_source()`` at the end to satisfy the
        ``BaseAcquisition`` contract.
        """
        if self._task_active:
            return

        self._ai_task.start()
        self._task_active = True
        super().set_data_source()

    def terminate_data_source(self) -> None:
        """Stop the NIDAQmx task.

        Safe to call multiple times; subsequent calls when the task is
        already stopped are silently ignored.
        """
        if not self._task_active:
            return

        self._ai_task.task.stop()
        self._task_active = False

    def read_data(self) -> np.ndarray:
        """Read all currently available samples from the device buffer.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_samples, n_channels)``.  Returns an
            empty array of shape ``(0, n_channels)`` when no data is
            available.
        """
        n_channels = len(self._channel_names_init)

        raw = self._ai_task.acquire(n_samples=None)

        if raw is None or raw.size == 0:
            return np.empty((0, n_channels))

        return raw

    def clear_buffer(self) -> None:
        """Read and discard all data currently in the device buffer."""
        self._ai_task.acquire(n_samples=None)
