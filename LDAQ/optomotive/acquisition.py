"""Acquisition sources for Optomotive (vevo) cameras.

The camera delivers two streams at the same time: stream 0 is an 8-bit
grayscale image and stream 1 is the FPGA output, whose byte layout depends
on the FPGA program. Both streams belong to the same physical camera, so
one :class:`OptomotiveCamera` object is shared by the two LDAQ sources
:class:`OptomotiveImage` (stream 0) and :class:`OptomotiveRaw` (stream 1).

Installation: ``vevoapi.py`` and ``liboptomotive_vevo.so`` (vendor files)
must be importable, for example placed in the folder of the virtual
environment and added to the Python path. Importing LDAQ works without them;
the classes raise ``ImportError`` when they are created.

Example::

    import LDAQ
    from LDAQ.optomotive import OptomotiveCamera, OptomotiveImage, OptomotiveRaw

    cam = OptomotiveCamera(roi=(0, 0, 640, 512), sample_rate=1000.0)
    img = OptomotiveImage(cam)
    raw = OptomotiveRaw(cam, frame_bits=32768)
    ldaq = LDAQ.Core(acquisitions=[img, raw])
    ldaq.run(measurement_duration=2.0, autostart=True, hotkeys=False)
    cam.close()
"""
import atexit
import ctypes
import threading
import time
import warnings

import numpy as np

from ..acquisition_base import BaseAcquisition

try:
    import vevoapi
except (ImportError, OSError):  # OSError: vevoapi loads the .so at import
    vevoapi = None

_STREAM_IMAGE = 0
_STREAM_RAW = 1
_ALL_STREAMS = (_STREAM_IMAGE, _STREAM_RAW)

# After a stop the camera keeps sending frames from its on-board memory.
# The flush waits until no frame arrives for _FLUSH_QUIET_S seconds,
# but at most _FLUSH_TIMEOUT_S seconds.
_FLUSH_QUIET_S = 0.2
_FLUSH_TIMEOUT_S = 10.0

_IMPORT_ERROR_MESSAGE = (
    "vevoapi could not be imported. vevoapi.py and liboptomotive_vevo.so "
    "must be importable, for example placed in the folder of the virtual "
    "environment and added to the Python path (.pth file).")

# OptomotiveCamera objects of this process that are not closed yet. A camera
# that is left open (an error before close(), a notebook cell that failed)
# does not answer a new connection, so it is closed before a new
# OptomotiveCamera is created and when Python exits.
_open_cameras = []
_open_cameras_lock = threading.Lock()


def _close_open_cameras(warn):
    """Close every OptomotiveCamera of this process that is still open.

    Args:
        warn (bool): Issue a RuntimeWarning for each camera that is closed.
    """
    with _open_cameras_lock:
        cameras = list(_open_cameras)
    for camera in cameras:
        if warn:
            warnings.warn(
                "An earlier OptomotiveCamera was still open. It was closed "
                "before the new one was created.", RuntimeWarning, stacklevel=3)
        try:
            camera.close()
        except Exception as e:  # keep closing the others
            print(f"OptomotiveCamera: close failed: {e!r}")


atexit.register(_close_open_cameras, False)


class OptomotiveCamera:
    """Shared handle of one Optomotive camera, used by both LDAQ sources.

    This is a plain class and not an LDAQ acquisition source. It opens the
    camera, applies the settings and starts/stops the image acquisition.
    The camera is started when the first source registers and stopped when
    the last source unregisters (reference counting). All vevo calls made
    during acquisition run under :attr:`lock`.

    Only one OptomotiveCamera can be open at a time. A camera that is still
    open when a new one is created is closed first, and every open camera is
    closed when Python exits. Calling :meth:`close` yourself is still best.

    Example::

        cam = OptomotiveCamera(roi=(0, 0, 640, 512), sample_rate=1000.0)
        img = OptomotiveImage(cam)
        raw = OptomotiveRaw(cam, frame_bits=32768)
        ldaq = LDAQ.Core(acquisitions=[img, raw])
        ldaq.run(measurement_duration=2.0, autostart=True, hotkeys=False)
        cam.close()

    Attributes:
        sample_rate (float): Actual frame rate read back from the camera.
        lock (threading.Lock): Lock for every vevo call during acquisition.
    """

    def __init__(self, vevo_camera=None, roi=None, sample_rate=None,
                 exposure=None, sdk_buffer_size=1000, discover_timeout=10.0):
        """Open (or wrap) the camera and apply the settings.

        Settings are applied in this order, because the frame-rate limit
        depends on exposure and ROI: exposure, ROI, frame rate. Each is
        applied only if it is not None.

        Before the settings are written, the acquisition is stopped and
        frames still sent by the camera are dropped (see :meth:`_flush`).

        Args:
            vevo_camera (vevoapi.Camera, optional): An already opened camera.
                If None, the first camera found by ``vevoapi.discover()`` is
                opened and owned by this object (closed in :meth:`close`).
                A passed-in camera is never closed by this object.
            roi (tuple, optional): ``(x, y, w, h)`` passed to
                ``Camera.set_roi``.
            sample_rate (float, optional): Requested frame rate in Hz. The
                actual rate is read back into :attr:`sample_rate`; a
                RuntimeWarning is issued if it differs by more than 1 %.
            exposure (float, optional): Passed unchanged to vevoapi
                ``Camera.set_exposure`` (register REG_EXPOSURETIME).
            sdk_buffer_size (int, optional): Number of frames of the SDK
                buffer (``vevoSetBufferSize``). Defaults to 1000.
            discover_timeout (float, optional): Used only if ``vevo_camera``
                is None. ``vevoapi.discover()`` is repeated until a camera
                answers, for at most this many seconds. After a fast run the
                camera does not answer for a short time after it is closed.
                Defaults to 10.0.

        Raises:
            ImportError: If vevoapi could not be imported.
            RuntimeError: If no camera is found, the camera cannot be opened,
                a setting cannot be written or the frame rate cannot be read.
        """
        if vevoapi is None:
            raise ImportError(_IMPORT_ERROR_MESSAGE)

        _close_open_cameras(warn=True)

        self.lock = threading.Lock()
        self._registered = {}  # id(source) -> stream number
        self._running = False
        self._closed = False
        self._owns_camera = vevo_camera is None
        self.cam = None
        self.sample_rate = None

        if self._owns_camera:
            t_end = time.monotonic() + discover_timeout
            while True:
                cameras = vevoapi.discover()
                if cameras:
                    break
                if time.monotonic() >= t_end:
                    raise RuntimeError(
                        f"No Optomotive camera found within "
                        f"{discover_timeout} s.")
                time.sleep(0.1)
            cam = cameras[0]
            if not cam.open():
                raise RuntimeError("Could not open the Optomotive camera.")
            self.cam = cam
        else:
            self.cam = vevo_camera

        try:
            # A previous session may have left the camera sending frames.
            # Stop it and drop those frames before writing any setting.
            self.cam.stop_image_acquisition()
            self._flush()

            if exposure is not None:
                if self.cam.set_exposure(exposure) is None:
                    raise RuntimeError(
                        f"Could not set the exposure to {exposure}.")
            if roi is not None:
                x, y, w, h = roi
                if any(r is None for r in self.cam.set_roi(x, y, w, h)):
                    raise RuntimeError(f"Could not set the ROI to {roi}.")
            if sample_rate is not None:
                if self.cam.set_framerate(sample_rate) is None:
                    raise RuntimeError(
                        f"Could not set the frame rate to {sample_rate}.")

            vevoapi.vevo_lib.vevoSetBufferSize(
                ctypes.c_uint32(self.cam.handle),
                ctypes.c_uint32(sdk_buffer_size))

            actual_rate = self.cam.get_framerate()
            if actual_rate is None:
                raise RuntimeError("Could not read the frame rate of the camera.")
            self.sample_rate = float(actual_rate)
        except BaseException:  # also Ctrl+C, so the camera is not left open
            self._release_camera()
            raise

        with _open_cameras_lock:
            _open_cameras.append(self)

        if sample_rate is not None and sample_rate > 0:
            rel_err = abs(self.sample_rate - sample_rate) / sample_rate
            if rel_err > 0.01:
                warnings.warn(
                    f"Requested sample_rate {sample_rate:.1f} Hz was changed "
                    f"by the camera to {self.sample_rate:.1f} Hz (limit for "
                    f"current exposure and ROI). LDAQ timestamps will "
                    f"reflect the achieved rate.",
                    RuntimeWarning, stacklevel=2)

    @property
    def image_shape(self):
        """tuple: Image shape ``(H, W)``, binning-aware."""
        w, h = self.cam.get_image_size()
        return (int(h), int(w))

    @property
    def is_running(self):
        """bool: True while the image acquisition is started."""
        return self._running

    def register(self, source, stream):
        """Register a source and start the acquisition if it is the first.

        Idempotent per source (tracked by ``id``). On the first registration
        both stream buffers are cleared before the acquisition is started.

        Args:
            source (object): The source that uses the camera.
            stream (int): Stream number used by the source (0 or 1).

        Raises:
            RuntimeError: If the camera is closed or does not start.
        """
        with self.lock:
            if self._closed:
                raise RuntimeError("OptomotiveCamera is closed.")
            if id(source) in self._registered:
                return
            if not self._registered:
                for s in _ALL_STREAMS:
                    self.cam.clear_buffer(s)
                if not self.cam.start_image_acquisition():
                    raise RuntimeError("Camera did not start the image acquisition.")
                self._running = True
            self._registered[id(source)] = stream

    def unregister(self, source):
        """Unregister a source and stop the acquisition if it was the last.

        Idempotent per source.

        Args:
            source (object): The source that was registered.
        """
        with self.lock:
            if id(source) not in self._registered:
                return
            del self._registered[id(source)]
            if not self._registered and self._running:
                self.cam.stop_image_acquisition()
                self._running = False

    def clear_stream(self, stream):
        """Clear the SDK buffer of one stream (under the lock).

        Args:
            stream (int): Stream number (0 or 1).
        """
        with self.lock:
            self.cam.clear_buffer(stream)

    def _copy_frame(self, image, n_bytes, skip_header, out_row):
        """Copy one frame once from SDK memory into ``out_row``.

        Args:
            image (vevoapi.Image): Frame returned by ``get_next_image``.
            n_bytes (int): Expected number of bytes (without header if
                ``skip_header``, otherwise the whole frame).
            skip_header (bool): Drop the header from the copy.
            out_row (np.ndarray): uint8 row of length ``n_bytes``.

        Returns:
            tuple: ``(frame_size, header_size, copied)``. ``copied`` is
            False (nothing written) if the size does not match ``n_bytes``.
        """
        lib = vevoapi.vevo_lib
        handle = ctypes.c_void_p(image.handle)
        ptr = lib.vevoGetImageData(handle)
        size = int(lib.vevoGetImageSize(handle))
        header = int(lib.vevoGetHeaderSize(handle))
        payload = size - header if skip_header else size
        if payload != n_bytes or not ptr:
            return size, header, False
        src = np.frombuffer((ctypes.c_uint8 * size).from_address(ptr),
                            dtype=np.uint8)
        start = header if skip_header else 0
        np.copyto(out_row, src[start:start + n_bytes])
        return size, header, True

    def read_stream(self, stream, n_bytes, skip_header):
        """Drain all pending frames of one stream.

        Frames are copied once from SDK memory into a preallocated array and
        every image is popped. Frames of the wrong size are skipped and
        counted. Streams without a registered source are cleared so that
        their buffers do not fill up.

        Args:
            stream (int): Stream number (0 or 1).
            n_bytes (int): Expected bytes per frame (without the header if
                ``skip_header`` is True, otherwise the whole frame).
            skip_header (bool): Drop the header from the output.

        Returns:
            tuple: ``(data, n_bad, first_bad, last_header_size)`` where
            ``data`` is a uint8 array of shape ``(n_ok, n_bytes)``,
            ``n_bad`` the number of skipped frames, ``first_bad`` is
            ``(frame_size, header_size)`` of the first skipped frame or None,
            and ``last_header_size`` the header size of the last good frame
            (None if there was none).
        """
        empty = np.empty((0, n_bytes), dtype=np.uint8)
        with self.lock:
            if self._closed:
                return empty, 0, None, None

            active = set(self._registered.values())
            for s in _ALL_STREAMS:
                if s not in active:
                    self.cam.clear_buffer(s)
            if stream not in active:
                return empty, 0, None, None

            n_pending = int(self.cam.get_pending_image_count(stream))
            if n_pending <= 0:
                return empty, 0, None, None

            out = np.empty((n_pending, n_bytes), dtype=np.uint8)
            n_ok = 0
            n_bad = 0
            first_bad = None
            last_header_size = None
            for _ in range(n_pending):
                image = self.cam.get_next_image(stream, 0)
                if image is None:
                    break
                try:
                    size, header, copied = self._copy_frame(
                        image, n_bytes, skip_header, out[n_ok])
                finally:
                    self.cam.pop_image(image)
                if copied:
                    n_ok += 1
                    last_header_size = header
                else:
                    n_bad += 1
                    if first_bad is None:
                        first_bad = (size, header)
            return out[:n_ok], n_bad, first_bad, last_header_size

    def _flush(self):
        """Drop frames that the camera still sends after a stop.

        After ``stop_image_acquisition`` the camera keeps sending frames
        from its on-board memory. Writing registers or closing the camera
        while it is still sending can make the camera stop responding.
        Both stream buffers are cleared until no frame arrives for
        ``_FLUSH_QUIET_S`` seconds, but at most ``_FLUSH_TIMEOUT_S`` seconds.

        Returns:
            bool: True if the camera went quiet, False on timeout.
        """
        now = time.monotonic()
        t_end = now + _FLUSH_TIMEOUT_S
        t_quiet = now + _FLUSH_QUIET_S
        while True:
            now = time.monotonic()
            if now >= t_quiet:
                return True
            if now >= t_end:
                warnings.warn(
                    f"The camera was still sending frames {_FLUSH_TIMEOUT_S} s "
                    f"after the stop.", RuntimeWarning, stacklevel=2)
                return False
            got_frames = False
            for s in _ALL_STREAMS:
                if int(self.cam.get_pending_image_count(s)) > 0:
                    got_frames = True
                self.cam.clear_buffer(s)
            if got_frames:
                t_quiet = time.monotonic() + _FLUSH_QUIET_S
            time.sleep(0.01)

    def close(self):
        """Stop the acquisition and close the camera if it is owned.

        Frames still sent by the camera after the stop are dropped before
        the camera is closed (see :meth:`_flush`). A camera passed in by the
        user is not closed. Safe to call twice.
        """
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self._registered.clear()
            try:
                if self._running:
                    try:
                        self.cam.stop_image_acquisition()
                    finally:
                        self._running = False
                self._flush()
            finally:
                self._release_camera()
                with _open_cameras_lock:
                    if self in _open_cameras:
                        _open_cameras.remove(self)

    def _release_camera(self):
        """Close the camera if owned (best effort)."""
        if self._owns_camera and self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class _OptomotiveStreamSource(BaseAcquisition):
    """Common code of the two stream sources (private)."""

    _stream = None
    _skip_header = True

    def _setup(self, camera, acquisition_name, default_name, channel_name,
               n_bytes, channel_shape):
        self.acquisition_name = (default_name if acquisition_name is None
                                 else acquisition_name)
        self.camera = camera
        self._n_bytes = n_bytes
        self._registered = False
        self.n_bad_frames = 0
        self.last_header_size = None
        self._bad_warned = False

        self.buffer_dtype = np.uint8  # must be set before set_trigger()
        self._channel_names_video_init = [channel_name]
        self._channel_shapes_video_init = [channel_shape]
        self.sample_rate = camera.sample_rate

        self.set_data_source(start_acquisition=False)
        self.set_trigger(1e20, 0, duration=1.0)

    def set_data_source(self, start_acquisition=True):
        """Build the channel layout and (optionally) start the camera.

        Idempotent. The camera is started when the first source registers.

        Args:
            start_acquisition (bool, optional): Register with the camera
                (which starts it if needed). If False, only the channel
                layout is built. Defaults to True.
        """
        if start_acquisition and not self._registered:
            self.camera.register(self, self._stream)
            self._registered = True
        super().set_data_source()

    def read_data(self):
        """Drain all pending frames of this stream.

        Returns:
            np.ndarray: uint8 array of shape ``(n_frames, n_columns)``;
            ``(0, n_columns)`` if no frame is pending.
        """
        data, n_bad, first_bad, last_header = self.camera.read_stream(
            self._stream, self._n_bytes, self._skip_header)
        if last_header is not None:
            self.last_header_size = last_header
        if n_bad:
            self.n_bad_frames += n_bad
            if not self._bad_warned:
                self._bad_warned = True
                size, header = first_bad
                warnings.warn(
                    f"{self.acquisition_name}: frame with unexpected size "
                    f"skipped (stream {self._stream}, frame size {size} B, "
                    f"header {header} B, expected {self._n_bytes} B "
                    f"{'without' if self._skip_header else 'including'} "
                    f"header). Further bad frames are only counted in "
                    f"n_bad_frames.",
                    RuntimeWarning, stacklevel=2)
        return data

    def terminate_data_source(self):
        """Unregister from the camera. Idempotent; never closes the camera."""
        if self._registered:
            self._registered = False
            self.camera.unregister(self)

    def clear_buffer(self):
        """Clear the SDK buffer of this stream."""
        self.camera.clear_stream(self._stream)

    def get_sample_rate(self):
        """Return the camera frame rate in Hz."""
        return self.sample_rate


class OptomotiveImage(_OptomotiveStreamSource):
    """Stream 0 of an Optomotive camera: 8-bit grayscale image.

    One video channel of shape ``(H, W)``, uint8. The frame header is
    removed. Create an :class:`OptomotiveCamera` first and pass it in.
    """

    _stream = _STREAM_IMAGE
    _skip_header = True

    def __init__(self, camera, acquisition_name=None, channel_name="image"):
        """Create the image source (the camera is not started here).

        Args:
            camera (OptomotiveCamera): Shared camera handle.
            acquisition_name (str, optional): Name of the source. Defaults
                to "OptomotiveImage".
            channel_name (str, optional): Name of the video channel.
                Defaults to "image".

        Raises:
            ImportError: If vevoapi could not be imported.
            TypeError: If ``camera`` is not an OptomotiveCamera.
        """
        if vevoapi is None:
            raise ImportError(_IMPORT_ERROR_MESSAGE)
        if not isinstance(camera, OptomotiveCamera):
            raise TypeError("camera must be an OptomotiveCamera.")
        super().__init__()
        height, width = camera.image_shape
        self._setup(camera, acquisition_name, "OptomotiveImage", channel_name,
                    height * width, (height, width))


class OptomotiveRaw(_OptomotiveStreamSource):
    """Stream 1 of an Optomotive camera: FPGA output (raw bytes).

    The byte layout of stream 1 depends on the FPGA program, so frames are
    stored as raw bytes, WHOLE frame including the header. One video channel
    of shape ``(1, frame_bytes)``, uint8; the measurement therefore gives
    ``meas['video'][i]`` with shape ``(n_frames, 1, frame_bytes)``.

    The optional ``parser`` is a callable ``parser(raw) -> np.ndarray`` that
    turns the raw bytes into meaningful values. It is NOT applied during
    acquisition and its output is NOT stored in the ring buffer (the buffer
    is uint8, parsed values would be truncated). Apply it afterwards with
    :meth:`parse`. The attribute :attr:`parser` can be reassigned at any
    time.

    Attributes:
        parser (callable or None): Parser, see above.
        last_header_size (int or None): Header size of the last good
            stream-1 frame.
        n_bad_frames (int): Number of skipped frames with a wrong size.
    """

    _stream = _STREAM_RAW
    _skip_header = False

    def __init__(self, camera, frame_bits, parser=None, acquisition_name=None,
                 channel_name="raw"):
        """Create the raw source (the camera is not started here).

        Args:
            camera (OptomotiveCamera): Shared camera handle.
            frame_bits (int): Size of one whole stream-1 frame in bits,
                including its header. Positive and divisible by 8.
            parser (callable, optional): ``parser(raw) -> np.ndarray`` with
                ``raw`` a uint8 array of shape ``(n_frames, frame_bytes)``;
                the result must have ``n_frames`` as first dimension.
            acquisition_name (str, optional): Name of the source. Defaults
                to "OptomotiveRaw".
            channel_name (str, optional): Name of the video channel.
                Defaults to "raw".

        Raises:
            ImportError: If vevoapi could not be imported.
            TypeError: If ``camera`` is not an OptomotiveCamera.
            ValueError: If ``frame_bits`` is not a positive int divisible
                by 8.
        """
        if vevoapi is None:
            raise ImportError(_IMPORT_ERROR_MESSAGE)
        if not isinstance(camera, OptomotiveCamera):
            raise TypeError("camera must be an OptomotiveCamera.")
        if (isinstance(frame_bits, bool) or not isinstance(frame_bits, (int, np.integer))
                or frame_bits <= 0 or frame_bits % 8 != 0):
            raise ValueError("frame_bits must be a positive int divisible by 8.")
        super().__init__()
        self.frame_bits = int(frame_bits)
        self.frame_bytes = self.frame_bits // 8
        self.parser = parser
        self._setup(camera, acquisition_name, "OptomotiveRaw", channel_name,
                    self.frame_bytes, (1, self.frame_bytes))

    def parse(self, raw=None):
        """Apply :attr:`parser` to raw frames.

        Args:
            raw (np.ndarray, optional): uint8 array of shape
                ``(n_frames, frame_bytes)``, or a measurement-dict video
                array of shape ``(n_frames, 1, frame_bytes)``. If None, the
                raw data recorded in this source's ring buffer is used.

        Returns:
            np.ndarray: Output of the parser, ``n_frames`` as first
            dimension.

        Raises:
            ValueError: If no parser is set or ``raw`` has a wrong shape.
        """
        if self.parser is None:
            raise ValueError(
                "No parser set. Pass parser=... to OptomotiveRaw or assign "
                "it to the 'parser' attribute.")
        if raw is None:
            raw = self.get_measurement_dict()['video'][0]
        raw = np.asarray(raw)
        if raw.ndim == 3 and raw.shape[1] == 1:
            raw = raw[:, 0, :]
        if raw.ndim != 2 or raw.shape[1] != self.frame_bytes:
            raise ValueError(
                f"raw must have shape (n_frames, {self.frame_bytes}) or "
                f"(n_frames, 1, {self.frame_bytes}), got {raw.shape}.")
        return self.parser(raw)
