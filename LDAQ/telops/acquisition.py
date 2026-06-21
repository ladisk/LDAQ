import warnings

import numpy as np

from ..acquisition_base import BaseAcquisition

try:
    from pyTelops import Camera as _PyTelopsCamera
except ImportError:
    _PyTelopsCamera = None


class TelopsCamera(BaseAcquisition):
    """
    Acquisition class for Telops thermal cameras over GigE Vision.

    Uses the pyTelops driver, a pure-Python GigE Vision implementation
    that does not require the Telops eBUS SDK. Tested against Telops
    FAST-series MWIR cameras (e.g. FAST M3k).

    Installation:

    - Install pyTelops:  ``pip install pyTelops``
    - No vendor SDK, no compiled extensions required.

    The plugin can either own its own :class:`pyTelops.Camera` instance
    (default) or wrap a pre-configured one passed via ``camera=``. The
    pre-configured pattern is useful when the camera needs lens-specific
    calibration loading, NUC, or other one-time setup before acquisition.

    Examples
    --------
    Standalone (plugin owns the camera)::

        import LDAQ
        tel = LDAQ.telops.TelopsCamera(sample_rate=100.0,
                                        integration_time=50.0)
        with tel:
            tel.run_acquisition(run_time=2.0)
            t, data = tel.get_data()

    Pre-configured camera (with calibration block already loaded)::

        from pyTelops import Camera
        cam = Camera()
        cam.connect()
        cam.calibration_load(lens="50mm", temp=25)

        tel = LDAQ.telops.TelopsCamera(sample_rate=100.0, camera=cam)
        ni  = LDAQ.national_instruments.NIAcquisition(...)

        ldaq = LDAQ.Core(acquisitions=[tel, ni])
        ldaq.run(measurement_duration=5.0)

        cam.disconnect()  # user owns the camera, must clean up
    """

    def __init__(self, acquisition_name=None, sample_rate=None,
                 channel_name="thermal_field",
                 calibration_mode=None,
                 integration_time=None,
                 packet_delay=None,
                 camera=None):
        """Initialize the Telops camera acquisition source.

        For an owned camera (``camera=None``, default), the plugin
        applies sensible defaults (calibration_mode="RT",
        sample_rate=100.0). For a user-provided camera (``camera=cam``),
        the plugin respects the camera's existing configuration unless
        the corresponding kwarg is explicitly passed.

        Args:
            acquisition_name (str, optional): Name of this acquisition
                source in the LDAQ Core. Defaults to "TelopsCamera".

            sample_rate (float, optional): Camera frame rate in Hz.
                If ``None`` (default), the plugin uses 100 Hz when it
                owns the camera, or whatever rate is currently set on
                a user-provided camera. The actual achieved rate may
                be clamped if the requested rate exceeds the camera's
                max for the current resolution and integration time;
                a warning is issued if the achieved rate differs by
                more than 1%. Check :attr:`sample_rate` after
                construction for the truthful value.

            channel_name (str, optional): Name of the thermal field
                video channel. Defaults to "thermal_field".

            calibration_mode (str, optional): Camera calibration mode.
                One of "RT" (radiometric temperature, Celsius float32),
                "NUC" (uint16 NUC counts), "RAW" (uint16 raw counts),
                "IBR", "IBI". If ``None`` (default), the plugin uses
                "RT" when it owns the camera, or respects the existing
                mode on a user-provided camera.

            packet_delay (int, optional): GVSP inter-packet delay in
                camera timer ticks (8 ns each). Spreads the per-frame
                packet burst over time so a slow or jittery
                consumer/host process can't overflow the UDP receive
                queue. ``1000`` = ~8 us between packets, safe up to
                ~400 fps, eliminates most "packets unrecoverable"
                warnings on live streaming. The value persists across
                stream restarts. ``None`` (default) leaves whatever is
                currently configured on the camera (usually 0 = no
                delay).

            integration_time (float, optional): Integration time in
                microseconds. If ``None`` (default), uses whatever is
                currently set on the camera.

            camera (pyTelops.Camera, optional): A pre-configured and
                connected pyTelops Camera instance. If ``None`` (default),
                the plugin creates and manages its own Camera, connecting
                in ``__init__`` and disconnecting in
                ``terminate_data_source``. If a pre-connected Camera is
                passed in, the user retains ownership and is responsible
                for calling ``cam.disconnect()`` after acquisition.

        Raises:
            ImportError: If pyTelops is not installed.
            RuntimeError: If a user-provided camera is not connected.
        """
        if _PyTelopsCamera is None:
            raise ImportError(
                "pyTelops is not installed. Install it with: "
                "pip install pyTelops")

        super().__init__()
        self.acquisition_name = ('TelopsCamera' if acquisition_name is None
                                 else acquisition_name)

        # Internal flags; initialize early so terminate_data_source is
        # safe to call even if __init__ raises mid-way.
        self._owns_camera = (camera is None)
        self.cam = None
        self.camera_acq_started = False
        self._acq_broken = False  # set by read_data on persistent failure

        try:
            # ----------------------------------------------------------
            # Camera ownership: create+connect, or wrap a pre-connected
            # ----------------------------------------------------------
            if self._owns_camera:
                self.cam = _PyTelopsCamera()
                self.cam.connect()
            else:
                self.cam = camera
                if not self.cam.is_connected:
                    raise RuntimeError(
                        "Pre-configured camera must be connected. Call "
                        "cam.connect() before passing it to TelopsCamera.")

            # ----------------------------------------------------------
            # Calibration mode: apply default only when plugin owns the
            # camera and user didn't pass one. For user cameras with
            # no kwarg, respect their pre-set configuration.
            # ----------------------------------------------------------
            if calibration_mode is None and self._owns_camera:
                calibration_mode = "RT"
            if calibration_mode is not None:
                self.cam.calibration_mode = calibration_mode

            # ----------------------------------------------------------
            # Integration time: only set if user explicitly requested
            # ----------------------------------------------------------
            if integration_time is not None:
                self.cam.integration_time = integration_time

            # ----------------------------------------------------------
            # Packet delay: spreads per-frame burst over time so a
            # slow or jittery consumer/host process can keep up.
            # Persists across stream restarts.
            # ----------------------------------------------------------
            if packet_delay is not None:
                self.cam.packet_delay = packet_delay

            # ----------------------------------------------------------
            # Sample rate: default 100 Hz only for owned cameras
            # ----------------------------------------------------------
            if sample_rate is None and self._owns_camera:
                sample_rate = 100.0
            if sample_rate is not None:
                # The driver's frame_rate setter emits its own
                # UserWarning when it clamps. Suppress it so only the
                # plugin's richer RuntimeWarning fires for one clamp event.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.cam.frame_rate = sample_rate
            self.sample_rate = self.cam.frame_rate

            # Warn if achieved rate differs significantly from requested
            if sample_rate is not None and sample_rate > 0:
                rel_err = abs(self.sample_rate - sample_rate) / sample_rate
                if rel_err > 0.01:
                    warnings.warn(
                        f"Requested sample_rate {sample_rate:.1f} Hz was "
                        f"clamped by the camera to {self.sample_rate:.1f} Hz "
                        f"(max for current resolution and integration time). "
                        f"LDAQ timestamps will reflect the achieved rate.",
                        RuntimeWarning, stacklevel=2)

            # ----------------------------------------------------------
            # Output dtype: float32 for calibrated/temperature modes,
            # uint16 for raw counts. Set BEFORE set_trigger() so the
            # ring buffer is allocated with the correct dtype.
            # ----------------------------------------------------------
            active_mode = self.cam.calibration_mode
            # Handle both enum (real pyTelops) and string (test mocks)
            mode_name = (active_mode.name if hasattr(active_mode, "name")
                         else str(active_mode)).upper()
            # dtype assumption: NUC/RAW frames carry zero header
            # coefficients, so the driver's convert step is a no-op cast
            # and the data stays uint16. The float modes (RT, IBR, IBI)
            # return float32. The astype in read_data relies on this.
            if mode_name in ("RT", "IBR", "IBI"):
                self.buffer_dtype = np.float32
            elif mode_name in ("NUC", "RAW", "RAW0"):
                self.buffer_dtype = np.uint16
            else:
                # Unknown mode; default to uint16 and warn
                warnings.warn(
                    f"Unknown calibration mode '{mode_name}'. "
                    f"Defaulting buffer_dtype to uint16.",
                    RuntimeWarning, stacklevel=2)
                self.buffer_dtype = np.uint16

            # ----------------------------------------------------------
            # Channel setup: single video channel of shape (H, W).
            # pyTelops returns (W, H); numpy convention is (rows, cols).
            # ----------------------------------------------------------
            self._channel_names_video_init = [channel_name]
            self._channel_shapes_video_init = []  # set in set_data_source

            self.set_data_source()
            self.set_trigger(1e20, 0, duration=1.0)

        except Exception:
            # Init failed; clean up so we don't leak the camera socket
            # or leave the GVSP receiver running.
            self._safe_cleanup()
            raise

    def _safe_cleanup(self):
        """Best-effort teardown of camera resources. Safe to call from
        a partially-initialized state."""
        if self.camera_acq_started:
            try:
                self.cam.acquisition_stop()
            except Exception:
                pass
            self.camera_acq_started = False
        if self._owns_camera and self.cam is not None:
            try:
                self.cam.disconnect()
            except Exception:
                pass

    def set_data_source(self):
        """Configure acquisition source. Idempotent, safe to call repeatedly.

        Reads the current camera resolution, populates the channel
        shape, and starts continuous acquisition if not already running.
        """
        # Populate channel shape from current camera resolution
        if not self._channel_shapes_video_init:
            w, h = self.cam.resolution
            self._channel_shapes_video_init = [(h, w)]  # numpy (rows, cols)

        # Start continuous acquisition if not already running
        if not self.camera_acq_started:
            self.cam.acquisition_start()
            self.camera_acq_started = True

        super().set_data_source()

    def read_data(self):
        """Pull the newest available frame from the running acquisition.

        Non-blocking: returns an empty array if no new frame is ready.
        Uses ``cam.read_frame(latest=True)`` to **drain** the driver's
        internal frame queue and return only the most recent frame;
        older queued frames are discarded. This bounds end-to-end
        latency when the consuming display/processing loop is slower
        than the camera, so the live preview stays in sync with real
        time instead of lagging further behind each second.

        Trade-off: if the consumer is slow, intermediate frames are
        dropped. Because stale frames are discarded (latest=True), the
        frame timestamps of a recorded measurement reflect the achieved
        consumer/poll rate, not the camera frame rate, whenever the
        consumer is slower than the camera. For faithful high-frame-rate
        recorded capture, use the onboard-buffer workflow instead (see
        example 014).

        Returns a 2D array of shape ``(n_samples, n_columns)`` where
        ``n_samples`` is 0 or 1 and ``n_columns = H * W``.

        If the underlying camera raises ``RuntimeError`` (acquisition
        stopped externally, e.g. network drop or forced disconnect),
        a warning is issued the first time and subsequent reads return
        empty without re-raising. LDAQ Core will then time out cleanly
        rather than spin forever on a dead camera.

        Returns:
            np.ndarray: Shape ``(0, H*W)`` if no frame ready,
                ``(1, H*W)`` if a frame was pulled.
        """
        shape = self.channel_shapes[
            self.channel_names_all.index(self.channel_names_video[0])]
        n_pixels = shape[0] * shape[1]

        try:
            # latest=True drains the queue and keeps only the newest
            # frame, bounding live-display latency when the consumer
            # is slower than the camera.
            frame = self.cam.read_frame(timeout=0.0, latest=True)
        except RuntimeError as e:
            if not self._acq_broken:
                warnings.warn(
                    f"TelopsCamera read_frame raised RuntimeError: {e}. "
                    f"Acquisition may have been stopped externally. "
                    f"Subsequent reads will return empty.",
                    RuntimeWarning, stacklevel=2)
                self._acq_broken = True
            return np.empty((0, n_pixels), dtype=self.buffer_dtype)

        if frame is None:
            return np.empty((0, n_pixels), dtype=self.buffer_dtype)

        # Cast to expected dtype (no-copy if already correct)
        return frame.reshape(1, n_pixels).astype(self.buffer_dtype, copy=False)

    def terminate_data_source(self):
        """Stop continuous acquisition and clean up.

        If the plugin owns the camera, the camera is disconnected.
        Otherwise only the acquisition is stopped; the camera remains
        connected and the user retains full ownership. Safe to call
        multiple times.
        """
        if getattr(self, "camera_acq_started", False):
            try:
                self.cam.acquisition_stop()
            except Exception:
                pass
            self.camera_acq_started = False

        if (getattr(self, "_owns_camera", False)
                and getattr(self, "cam", None) is not None):
            try:
                if self.cam.is_connected:
                    self.cam.disconnect()
            except Exception:
                pass

    def get_sample_rate(self):
        """Return the camera's sample (frame) rate in Hz."""
        return self.sample_rate

    def clear_buffer(self):
        """Drain any pending frames from the camera buffer.

        Pulls frames non-blocking until the queue is empty, capped at
        a reasonable upper bound to prevent infinite loops if the
        producer outpaces the drain.
        """
        if not self.camera_acq_started:
            return
        try:
            for _ in range(10000):  # cap to avoid infinite drain
                frame = self.cam.read_frame(timeout=0.0)
                if frame is None:
                    break
        except RuntimeError:
            pass
