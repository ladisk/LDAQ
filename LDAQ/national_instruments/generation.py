from __future__ import annotations

import numpy as np

try:
    import nidaqwrapper
    from nidaqwrapper import AOTask
    _NIDAQWRAPPER_AVAILABLE = True
except ImportError:
    _NIDAQWRAPPER_AVAILABLE = False

from ..generation_base import BaseGeneration


class NIGeneration(BaseGeneration):
    """
    Signal generation for NI analog output devices via nidaqwrapper.AOTask.

    This is a thin wrapper: channel configuration and timing are owned by the
    AOTask object passed in. NIGeneration is responsible only for lifecycle
    management (start, generate, safe-stop) and signal storage.

    Parameters
    ----------
    task : AOTask | str
        Either a configured ``nidaqwrapper.AOTask`` instance, or the name of
        an NI MAX task (str) which is resolved at construction time via
        ``nidaqwrapper.get_task_by_name()``.
    signal : numpy.ndarray | None, optional
        Signal to output. Shape ``(n_samples, n_channels)`` or ``(n_samples,)``
        for a single channel. Can also be set later with
        ``set_generation_signal()``. Default is None.
    generation_name : str | None, optional
        Human-readable name for this generation source. Defaults to the task
        name reported by the AOTask.

    Raises
    ------
    ImportError
        If ``nidaqwrapper`` is not installed.
    TypeError
        If ``task`` is neither an ``AOTask`` instance nor a ``str``.
    ValueError
        If a string task name cannot be resolved to an NI MAX task.

    Examples
    --------
    >>> ao = AOTask("my_ao_task", sample_rate=10_000)
    >>> ao.add_channel(...)
    >>> gen = NIGeneration(ao, signal=my_signal)

    >>> gen = NIGeneration("MyNIMaxTask", signal=my_signal)
    """

    def __init__(
        self,
        task: AOTask | str,
        signal: np.ndarray | None = None,
        generation_name: str | None = None,
    ) -> None:
        if not _NIDAQWRAPPER_AVAILABLE:
            raise ImportError(
                "nidaqwrapper is not installed. "
                "Install it before using NIGeneration."
            )

        super().__init__()

        if isinstance(task, AOTask):
            self._ao_task: AOTask = task
        elif isinstance(task, str):
            ni_task = nidaqwrapper.get_task_by_name(task)
            if ni_task is None:
                raise ValueError(
                    f"NI MAX task '{task}' not found. "
                    "Verify the task name in NI MAX."
                )
            self._ao_task = AOTask.from_task(ni_task)
        else:
            raise TypeError(
                f"task must be an AOTask instance or a str task name, "
                f"got {type(task).__name__!r}."
            )

        self.generation_name = (
            generation_name
            if generation_name is not None
            else self._ao_task.task_name
        )

        # Internal flag so terminate_data_source() is idempotent.
        self._task_active: bool = False

        # Cache channel count at construction so terminate_data_source() does
        # not rely on the attribute name being accessible at shutdown time.
        self._n_channels: int = self._ao_task.number_of_ch

        # Populated by set_generation_signal(); used to size the zeros burst
        # in terminate_data_source() even if self.signal is later cleared.
        self._n_samples_signal: int = 10  # safe fallback before any signal is set

        self.signal: np.ndarray | None = None
        if signal is not None:
            self.set_generation_signal(signal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_generation_signal(self, signal: np.ndarray) -> None:
        """
        Store the signal that will be written on each ``generate()`` call.

        nidaqmx expects shape ``(n_channels, n_samples)`` for multi-channel
        output. If a 2-D array with shape ``(n_samples, n_channels)`` is
        provided it is transposed automatically. 1-D arrays are kept as-is.

        Parameters
        ----------
        signal : numpy.ndarray
            Output signal. Shape ``(n_samples, n_channels)`` or
            ``(n_samples,)`` for a single channel.
        """
        if signal.ndim == 2:
            # Convert from (n_samples, n_channels) → (n_channels, n_samples).
            self.signal = signal.T
        else:
            self.signal = signal
        # Record sample count from the user-facing shape (first dim is always
        # n_samples in the input convention) so terminate can replicate it.
        self._n_samples_signal = signal.shape[0]

    def set_data_source(self) -> None:
        """
        Start the AOTask so it is ready to accept ``generate()`` calls.

        Safe to call multiple times; subsequent calls are no-ops while the
        task is already active.
        """
        if not self._task_active:
            self._ao_task.start(start_task=True)
            self._task_active = True

    def generate(self) -> None:
        """
        Write the stored signal to the analog output hardware.

        Raises
        ------
        ValueError
            If no signal has been set via the constructor or
            ``set_generation_signal()``.
        """
        if self.signal is None:
            raise ValueError(
                "No signal set. Call set_generation_signal() before generate()."
            )
        self._ao_task.generate(self.signal)

    def terminate_data_source(self) -> None:
        """
        Write zeros to all output channels, then stop the NI task.

        Writing zeros first ensures the physical outputs are left at a safe
        0 V state rather than holding the last generated value. The task
        handle is kept alive (``stop()`` rather than ``clear_task()``) so
        that ``set_data_source()`` can restart it in a subsequent cycle.
        Idempotent: calling this method on an already-terminated task is a
        no-op.
        """
        if not self._task_active:
            return

        # Drive outputs to 0 V before stopping. Use cached counts so this
        # does not depend on any mutable state being consistent at shutdown.
        if self._n_channels > 1:
            zeros = np.zeros((self._n_channels, self._n_samples_signal))
        else:
            zeros = np.zeros(self._n_samples_signal)
        self._ao_task.generate(zeros)

        # Stop without destroying the handle so the task can be restarted.
        self._ao_task.task.stop()
        self._task_active = False
