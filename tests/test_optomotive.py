"""Tests for the Optomotive camera sources (no hardware required).

vevoapi is replaced by a fake module and the cameras by FakeCamera, which
has controllable pending counts, frame sizes and header sizes.
"""
import threading
import time
import warnings
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import LDAQ
from LDAQ.optomotive import acquisition as acq_module
from LDAQ.optomotive import OptomotiveCamera, OptomotiveImage, OptomotiveRaw

WIDTH = 8
HEIGHT = 6
HEADER = 8
RAW_BYTES = 32


class FakeImage:
    def __init__(self, stream_nr, data, header_size):
        self.handle = 1
        self.stream_nr = stream_nr
        self.data = data
        self.header_size = header_size


class FakeCamera:
    """Quacks like vevoapi.Camera. Frames are queued with push()."""

    def __init__(self, width=WIDTH, height=HEIGHT, rate=1000.0):
        self.handle = 42
        self.width = width
        self.height = height
        self.rate = rate
        self.rate_after_set = None
        self.open_result = True
        self.calls = []
        self.queues = {0: deque(), 1: deque()}
        self.popped = 0
        self.producer = None  # optional callable(self) -> called on pending count
        self.running = False
        self.set_fails = None  # name of a setter that returns None (error)
        self.exposure = 1000.0
        # Optional frame-rate limits like the firmware: sensor readout limit
        # and 1e6 / (exposure + FOT_US). None: no limits.
        self.sensor_max = None

    def open(self):
        self.calls.append("open")
        return self.open_result

    def close(self):
        self.calls.append("close")
        return True

    FOT_US = 100.0

    def set_exposure(self, v):
        self.calls.append(("set_exposure", v))
        if self.set_fails == "set_exposure":
            return None
        self.exposure = v
        if self.sensor_max is not None:  # exposure caps the frame rate
            self.rate = min(self.rate, 1e6 / (v + self.FOT_US))
        return v

    def get_exposure(self):
        return self.exposure

    def set_roi(self, x, y, w, h):
        self.calls.append(("set_roi", x, y, w, h))
        if self.set_fails == "set_roi":
            return [x, y, None, h]
        self.width, self.height = w, h
        return [x, y, w, h]

    def set_framerate(self, v):
        self.calls.append(("set_framerate", v))
        if self.set_fails == "set_framerate":
            return None
        self.rate = v if self.rate_after_set is None else self.rate_after_set
        if self.sensor_max is not None:  # cap, then shorten the exposure
            self.rate = min(self.rate, self.sensor_max)
            self.exposure = min(self.exposure, 1e6 / self.rate - self.FOT_US)
        return self.rate

    def get_framerate(self):
        return self.rate

    def get_image_size(self):
        return [self.width, self.height]

    def start_image_acquisition(self):
        self.calls.append("start")
        self.running = True
        return True

    def stop_image_acquisition(self):
        self.calls.append("stop")
        self.running = False
        return True

    def clear_buffer(self, stream):
        self.calls.append(("clear", stream))
        self.queues[stream].clear()

    def get_pending_image_count(self, stream):
        if self.producer is not None:
            self.producer(self)
        return len(self.queues[stream])

    def get_next_image(self, stream, timeout_ms=0):
        if self.queues[stream]:
            return self.queues[stream].popleft()
        return None

    def pop_image(self, img):
        self.popped += 1

    def push(self, stream, size, header_size=HEADER, fill=1, timestamp=None):
        data = np.full(size, fill, dtype=np.uint8)
        if timestamp is not None:  # header bytes 0-7, little-endian
            data[:8] = np.frombuffer(np.uint64(timestamp).astype("<u8").tobytes(),
                                     dtype=np.uint8)
        self.queues[stream].append(FakeImage(stream, data, header_size))


_REAL_COPY_FRAME = OptomotiveCamera._copy_frame


def _fake_copy_frame(self, image, n_bytes, skip_header, out_row):
    size = image.data.size
    header = image.header_size
    payload = size - header if skip_header else size
    if payload != n_bytes:
        return size, header, False, None
    start = header if skip_header else 0
    out_row[:] = image.data[start:start + n_bytes]
    timestamp = int(image.data[:8].view("<u8")[0]) if header == 8 else None
    return size, header, True, timestamp


@pytest.fixture
def fake_vevoapi():
    """Patch the module-level vevoapi name; discover() returns one camera."""
    fake = MagicMock()
    fake.discover.return_value = [FakeCamera()]
    with patch.object(acq_module, "vevoapi", fake, create=True), \
            patch.object(OptomotiveCamera, "_copy_frame", _fake_copy_frame), \
            patch.object(acq_module, "_FLUSH_QUIET_S", 0.02), \
            patch.object(acq_module, "_FLUSH_TIMEOUT_S", 0.5):
        acq_module._open_cameras.clear()
        yield fake
        acq_module._open_cameras.clear()


@pytest.fixture
def fake_cam():
    return FakeCamera()


@pytest.fixture
def camera(fake_vevoapi, fake_cam):
    cam = OptomotiveCamera(vevo_camera=fake_cam)
    yield cam
    cam.close()


def _image_bytes():
    return WIDTH * HEIGHT + HEADER


# -----------------------------------------------------------------------------
# ImportError / import
# -----------------------------------------------------------------------------

class TestWithoutVevoapi:
    def test_import_ldaq_works(self):
        assert hasattr(LDAQ, "optomotive")

    def test_constructors_raise_import_error(self):
        with patch.object(acq_module, "vevoapi", None, create=True):
            with pytest.raises(ImportError, match="liboptomotive_vevo.so"):
                OptomotiveCamera()
            with pytest.raises(ImportError):
                OptomotiveImage(object())
            with pytest.raises(ImportError):
                OptomotiveRaw(object(), frame_bits=256)


# -----------------------------------------------------------------------------
# OptomotiveCamera
# -----------------------------------------------------------------------------

class TestOptomotiveCamera:
    def test_settings_order(self, fake_vevoapi, fake_cam):
        cam = OptomotiveCamera(vevo_camera=fake_cam, roi=(1, 2, 16, 12),
                               sample_rate=500.0, exposure=100.0)
        settings = [c[0] for c in fake_cam.calls
                    if isinstance(c, tuple) and c[0].startswith("set_")]
        assert settings == ["set_exposure", "set_roi", "set_framerate"]
        assert ("set_roi", 1, 2, 16, 12) in fake_cam.calls
        fake_vevoapi.vevo_lib.vevoSetBufferSize.assert_called_once()
        assert cam.image_shape == (12, 16)
        cam.close()

    def test_none_leaves_settings_untouched(self, fake_vevoapi, fake_cam):
        cam = OptomotiveCamera(vevo_camera=fake_cam)
        assert not [c for c in fake_cam.calls
                    if isinstance(c, tuple) and c[0].startswith("set_")]
        assert cam.sample_rate == 1000.0
        cam.close()

    def test_rate_readback_and_warning(self, fake_vevoapi, fake_cam):
        fake_cam.rate_after_set = 800.0
        with pytest.warns(RuntimeWarning, match="800.0"):
            cam = OptomotiveCamera(vevo_camera=fake_cam, sample_rate=1000.0)
        assert cam.sample_rate == 800.0
        cam.close()

    def test_no_warning_within_one_percent(self, fake_vevoapi, fake_cam, recwarn):
        fake_cam.rate_after_set = 995.0
        cam = OptomotiveCamera(vevo_camera=fake_cam, sample_rate=1000.0)
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        cam.close()

    def test_owned_camera_is_discovered_opened_and_closed(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]
        cam = OptomotiveCamera()
        assert "open" in owned.calls
        cam.close()
        cam.close()
        assert owned.calls.count("close") == 1

    def test_no_camera_found(self, fake_vevoapi):
        fake_vevoapi.discover.return_value = []
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="No Optomotive camera"):
            OptomotiveCamera(discover_timeout=0.3)
        assert time.monotonic() - t0 >= 0.3
        assert fake_vevoapi.discover.call_count > 1

    def test_camera_found_on_later_discover(self, fake_vevoapi):
        owned = FakeCamera()
        fake_vevoapi.discover.side_effect = [[], [], [owned]]
        cam = OptomotiveCamera(discover_timeout=5.0)
        assert fake_vevoapi.discover.call_count == 3
        assert "open" in owned.calls
        cam.close()

    def test_open_fails(self, fake_vevoapi):
        fake_vevoapi.discover.return_value[0].open_result = False
        with pytest.raises(RuntimeError, match="open"):
            OptomotiveCamera()

    def test_passed_in_camera_not_closed(self, fake_vevoapi, fake_cam):
        cam = OptomotiveCamera(vevo_camera=fake_cam)
        cam.close()
        cam.close()
        assert "close" not in fake_cam.calls

    def test_context_manager(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]
        with OptomotiveCamera() as cam:
            assert isinstance(cam, OptomotiveCamera)
        assert owned.calls.count("close") == 1

    def test_start_and_stop_once_with_two_sources(self, camera, fake_cam):
        n0 = len(fake_cam.calls)  # calls made by __init__
        a, b = object(), object()
        camera.register(a, 0)
        camera.register(a, 0)  # idempotent
        camera.register(b, 1)
        assert fake_cam.calls.count("start") == 1
        # buffers cleared before the start
        assert fake_cam.calls[n0:n0 + 3] == [("clear", 0), ("clear", 1), "start"]
        camera.unregister(a)
        camera.unregister(a)
        assert "stop" not in fake_cam.calls[n0:]
        camera.unregister(b)
        assert fake_cam.calls[n0:].count("stop") == 1
        assert not camera.is_running

    def test_close_stops_running_camera(self, fake_vevoapi, fake_cam):
        cam = OptomotiveCamera(vevo_camera=fake_cam)
        n0 = len(fake_cam.calls)
        cam.register(object(), 0)
        cam.close()
        assert fake_cam.calls[n0:].count("stop") == 1

    def test_init_stops_and_flushes_before_settings(self, fake_vevoapi, fake_cam):
        fake_cam.push(0, _image_bytes())  # left over from a previous session
        fake_cam.push(1, RAW_BYTES)
        cam = OptomotiveCamera(vevo_camera=fake_cam, exposure=100.0)
        i_set = fake_cam.calls.index(("set_exposure", 100.0))
        assert fake_cam.calls[0] == "stop"
        assert ("clear", 0) in fake_cam.calls[:i_set]
        assert ("clear", 1) in fake_cam.calls[:i_set]
        assert not fake_cam.queues[0] and not fake_cam.queues[1]
        cam.close()

    @pytest.mark.parametrize("setter", ["set_exposure", "set_roi", "set_framerate"])
    def test_failed_setting_raises_and_closes(self, fake_vevoapi, setter):
        owned = fake_vevoapi.discover.return_value[0]
        owned.set_fails = setter
        with pytest.raises(RuntimeError, match="Could not set"):
            OptomotiveCamera(roi=(0, 0, 16, 12), sample_rate=500.0,
                             exposure=100.0)
        assert owned.calls.count("close") == 1

    def test_new_camera_closes_one_left_open(self, fake_vevoapi):
        old_fake, new_fake = FakeCamera(), FakeCamera()
        fake_vevoapi.discover.side_effect = [[old_fake], [new_fake]]
        old = OptomotiveCamera()
        old.register(object(), 0)  # e.g. a run that failed before close()
        with pytest.warns(RuntimeWarning, match="still open"):
            new = OptomotiveCamera()
        assert old_fake.calls.count("close") == 1
        assert "stop" in old_fake.calls[old_fake.calls.index("start"):]
        # the old camera was closed before the new one was discovered
        assert fake_vevoapi.discover.call_count == 2
        assert acq_module._open_cameras == [new]
        new.close()
        assert acq_module._open_cameras == []

    def test_no_warning_when_previous_camera_was_closed(self, fake_vevoapi, recwarn):
        OptomotiveCamera().close()
        OptomotiveCamera().close()
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_exit_handler_closes_open_cameras(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]
        cam = OptomotiveCamera()
        cam.register(object(), 0)
        acq_module._close_open_cameras(False)  # what atexit calls
        assert owned.calls.count("close") == 1
        assert not cam.is_running
        assert acq_module._open_cameras == []

    def test_ctrl_c_during_init_closes_camera(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]

        def interrupted(v):
            raise KeyboardInterrupt

        owned.set_framerate = interrupted
        with pytest.raises(KeyboardInterrupt):
            OptomotiveCamera(sample_rate=500.0)
        assert owned.calls.count("close") == 1
        assert acq_module._open_cameras == []

    @pytest.mark.parametrize("exposure, expected", [(1900.0, 500.0), (900.0, 600.0)])
    def test_max_frame_rate_read_and_settings_restored(self, fake_vevoapi, exposure,
                                                       expected):
        fake = FakeCamera()
        fake.sensor_max = 600.0  # exposure 1900 us -> 1e6 / 2000 = 500 Hz
        cam = OptomotiveCamera(vevo_camera=fake, exposure=exposure, sample_rate=100.0)
        n0 = len(fake.calls)
        limits = cam.get_max_frame_rate()
        assert limits["sensor"] == pytest.approx(600.0)
        assert limits["with_exposure"] == pytest.approx(expected)
        assert ("set_framerate", acq_module._PROBE_FRAME_RATE) in fake.calls[n0:]
        assert fake.exposure == exposure
        assert fake.rate == 100.0
        assert cam.sample_rate == 100.0
        cam.close()

    def test_max_frame_rate_refused_while_running(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        n0 = len(fake_cam.calls)
        with pytest.raises(RuntimeError, match="acquiring"):
            camera.get_max_frame_rate()
        assert fake_cam.calls[n0:] == []
        img.terminate_data_source()

    def test_max_frame_rate_refused_when_closed(self, camera):
        camera.close()
        with pytest.raises(RuntimeError, match="closed"):
            camera.get_max_frame_rate()

    def test_max_frame_rate_restore_failure_raises(self, camera, fake_cam):
        fake_cam.sensor_max = 600.0
        fake_cam.set_fails = "set_exposure"
        with pytest.raises(RuntimeError, match="restore"):
            camera.get_max_frame_rate()

    def test_close_waits_until_camera_is_quiet(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]
        cam = OptomotiveCamera()
        cam.register(object(), 0)
        state = {"left": 5}

        def producer(c):  # frames keep arriving for a while after the stop
            if not c.running and "close" not in c.calls and state["left"] > 0:
                c.push(0, _image_bytes())
                state["left"] -= 1

        owned.producer = producer
        cam.close()
        assert state["left"] == 0
        assert owned.calls.index("close") > max(
            i for i, c in enumerate(owned.calls) if c == ("clear", 0))
        assert owned.calls.count("close") == 1

    def test_close_flush_timeout_warns(self, fake_vevoapi):
        owned = fake_vevoapi.discover.return_value[0]
        cam = OptomotiveCamera()
        owned.producer = lambda c: c.push(0, _image_bytes())  # never quiet
        with pytest.warns(RuntimeWarning, match="still sending"):
            cam.close()
        assert owned.calls.count("close") == 1

    def test_stream_without_source_is_cleared(self, camera, fake_cam):
        camera.register(object(), 0)
        fake_cam.push(1, 40)
        fake_cam.push(1, 40)
        fake_cam.calls.clear()
        data, n_bad, _, _, _ = camera.read_stream(0, WIDTH * HEIGHT, True)
        assert ("clear", 1) in fake_cam.calls
        assert ("clear", 0) not in fake_cam.calls
        assert len(fake_cam.queues[1]) == 0
        assert data.shape == (0, WIDTH * HEIGHT)

    def test_every_image_is_popped(self, camera, fake_cam):
        camera.register(object(), 0)
        for _ in range(3):
            fake_cam.push(0, _image_bytes())
        fake_cam.push(0, 5)  # bad
        data, n_bad, first_bad, last_header, _ = camera.read_stream(0, WIDTH * HEIGHT, True)
        assert data.shape == (3, WIDTH * HEIGHT)
        assert n_bad == 1
        assert first_bad == (5, HEADER)
        assert last_header == HEADER
        assert fake_cam.popped == 4


# -----------------------------------------------------------------------------
# OptomotiveImage
# -----------------------------------------------------------------------------

class TestOptomotiveImage:
    def test_layout_and_no_start_in_init(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        assert img.acquisition_name == "OptomotiveImage"
        assert img.channel_names_video == ["image"]
        assert img.channel_shapes == [(HEIGHT, WIDTH)]
        assert img.buffer_dtype == np.uint8
        assert img.sample_rate == 1000.0
        assert "start" not in fake_cam.calls
        assert img.get_sample_rate() == 1000.0

    def test_rejects_wrong_camera(self, fake_vevoapi):
        with pytest.raises(TypeError):
            OptomotiveImage(object())

    def test_drains_frames_and_removes_header(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        img.set_data_source()  # idempotent
        assert fake_cam.calls.count("start") == 1
        for i in range(4):
            fake_cam.push(0, _image_bytes(), fill=i + 1)
        data = img.read_data()
        assert data.shape == (4, WIDTH * HEIGHT)
        assert data.dtype == np.uint8
        assert list(data[:, 0]) == [1, 2, 3, 4]
        assert img.read_data().shape == (0, WIDTH * HEIGHT)
        assert img.last_header_size == HEADER
        img.terminate_data_source()

    def test_bad_frames_skipped_counted_warned_once(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        fake_cam.push(0, _image_bytes())
        fake_cam.push(0, 100)
        fake_cam.push(0, 100)
        with pytest.warns(RuntimeWarning, match="expected 48"):
            data = img.read_data()
        assert data.shape == (1, WIDTH * HEIGHT)
        assert img.n_bad_frames == 2
        fake_cam.push(0, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            img.read_data()
        assert img.n_bad_frames == 3
        img.terminate_data_source()

    def test_terminate_idempotent_and_does_not_close(self, camera, fake_cam):
        n0 = len(fake_cam.calls)  # calls made by __init__
        img = OptomotiveImage(camera)
        img.set_data_source()
        img.terminate_data_source()
        img.terminate_data_source()
        assert fake_cam.calls[n0:].count("stop") == 1
        assert "close" not in fake_cam.calls

    def test_clear_buffer(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        fake_cam.push(0, _image_bytes())
        img.clear_buffer()
        assert len(fake_cam.queues[0]) == 0
        img.terminate_data_source()


class TestDroppedFrames:
    PERIOD = 10_000  # ticks of the 10 MHz clock at 1000 fps

    def _push(self, fake_cam, frame_numbers, t0=123_456_789, jitter=0):
        for i, n in enumerate(frame_numbers):
            j = jitter if i % 2 else -jitter
            fake_cam.push(0, _image_bytes(), timestamp=t0 + n * self.PERIOD + j)

    def test_no_gap(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        self._push(fake_cam, range(10), jitter=300)
        img.read_data()
        self._push(fake_cam, range(10, 20), jitter=300)  # second read
        img.read_data()
        assert img.frame_timestamps.dtype == np.uint64
        assert img.dropped_frames() == {"n_frames": 20, "n_missing": 0, "gaps": []}
        img.terminate_data_source()

    def test_gap_of_three_frames(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        self._push(fake_cam, [0, 1, 2, 3, 4, 8, 9, 10], jitter=300)
        img.read_data()
        assert img.dropped_frames() == {"n_frames": 8, "n_missing": 3,
                                        "gaps": [(5, 3)]}
        img.terminate_data_source()

    def test_clear_and_new_run_reset_timestamps(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        self._push(fake_cam, range(3))
        img.read_data()
        img.clear_buffer()  # frames dropped on purpose are not "lost"
        self._push(fake_cam, range(100, 105))
        img.read_data()
        assert img.dropped_frames()["n_frames"] == 5
        assert img.dropped_frames()["n_missing"] == 0
        img.terminate_data_source()
        img.set_data_source()  # new run
        assert img.dropped_frames()["n_frames"] == 0
        img.terminate_data_source()

    def test_frame_without_timestamp_header(self, camera, fake_cam):
        img = OptomotiveImage(camera)
        img.set_data_source()
        fake_cam.push(0, WIDTH * HEIGHT + 4, header_size=4)
        img.read_data()
        assert img.frame_timestamps is None
        with pytest.raises(RuntimeError, match="timestamp header"):
            img.dropped_frames()
        img.terminate_data_source()

    def test_raw_stream_reads_no_timestamps(self, camera, fake_cam):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8)
        raw.set_data_source()
        fake_cam.push(1, RAW_BYTES)
        assert raw.read_data().shape == (1, RAW_BYTES)
        assert not hasattr(raw, "dropped_frames")
        raw.terminate_data_source()

    def test_real_copy_frame_reads_little_endian_timestamp(self, fake_vevoapi):
        frame = np.arange(8 + 5, dtype=np.uint8)
        frame[:8] = np.frombuffer((0x0102030405060708).to_bytes(8, "little"),
                                  dtype=np.uint8)
        lib = fake_vevoapi.vevo_lib
        lib.vevoGetImageData.return_value = frame.ctypes.data
        lib.vevoGetImageSize.return_value = frame.size
        lib.vevoGetHeaderSize.return_value = 8
        out = np.zeros(5, dtype=np.uint8)
        image = FakeImage(0, frame, 8)
        result = _REAL_COPY_FRAME(None, image, 5, True, out)
        assert result == (13, 8, True, 0x0102030405060708)
        assert list(out) == [8, 9, 10, 11, 12]
        lib.vevoGetHeaderSize.return_value = 0  # stream 1: no timestamp
        out = np.zeros(13, dtype=np.uint8)
        assert _REAL_COPY_FRAME(None, image, 13, False, out)[3] is None


# -----------------------------------------------------------------------------
# OptomotiveRaw
# -----------------------------------------------------------------------------

class TestOptomotiveRaw:
    @pytest.mark.parametrize("bits", [0, -8, 12, 33, 8.0, "256", True])
    def test_frame_bits_validation(self, camera, bits):
        with pytest.raises(ValueError):
            OptomotiveRaw(camera, frame_bits=bits)

    def test_layout(self, camera, fake_cam):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8)
        assert raw.acquisition_name == "OptomotiveRaw"
        assert raw.channel_names_video == ["raw"]
        assert raw.channel_shapes == [(1, RAW_BYTES)]
        assert raw.buffer_dtype == np.uint8
        assert "start" not in fake_cam.calls

    def test_header_kept_bad_sizes_skipped(self, camera, fake_cam):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8)
        raw.set_data_source()
        fake_cam.push(1, RAW_BYTES, header_size=8, fill=7)
        fake_cam.push(1, RAW_BYTES + 4, header_size=8)
        fake_cam.push(1, RAW_BYTES, header_size=16, fill=9)
        with pytest.warns(RuntimeWarning, match="expected 32"):
            data = raw.read_data()
        assert data.shape == (2, RAW_BYTES)
        assert data.dtype == np.uint8
        assert data[0, 0] == 7 and data[1, 0] == 9
        assert raw.n_bad_frames == 1
        assert raw.last_header_size == 16
        raw.terminate_data_source()

    def test_parse_with_parser_and_reassign(self, camera):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8,
                            parser=lambda r: r[:, 0].astype(np.int32))
        frames = np.arange(3 * RAW_BYTES, dtype=np.uint8).reshape(3, RAW_BYTES)
        out = raw.parse(frames)
        assert out.shape == (3,)
        out3d = raw.parse(frames.reshape(3, 1, RAW_BYTES))
        assert np.array_equal(out, out3d)
        raw.parser = lambda r: r[:, 1].astype(np.int32) * 2
        assert raw.parse(frames)[0] == 2

    def test_parse_wrong_shape(self, camera):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8, parser=lambda r: r)
        with pytest.raises(ValueError, match="shape"):
            raw.parse(np.zeros((3, RAW_BYTES + 1), dtype=np.uint8))

    def test_parse_without_parser(self, camera):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8)
        with pytest.raises(ValueError, match="No parser"):
            raw.parse(np.zeros((1, RAW_BYTES), dtype=np.uint8))

    def test_parse_from_ring_buffer(self, camera, fake_cam):
        raw = OptomotiveRaw(camera, frame_bits=RAW_BYTES * 8,
                            parser=lambda r: r[:, 0].astype(np.int32))

        def producer(c):
            if not c.running:
                return
            for _ in range(10):
                c.push(1, RAW_BYTES, fill=3)

        fake_cam.producer = producer
        raw.run_acquisition(run_time=0.05)  # 50 samples at 1000 Hz
        out = raw.parse()
        assert out.shape == (50,)
        assert np.all(out == 3)


# -----------------------------------------------------------------------------
# Core run
# -----------------------------------------------------------------------------

class TestCoreRun:
    def test_core_run_with_both_sources(self, fake_vevoapi):
        rate = 200.0
        cam_fake = FakeCamera(rate=rate)
        state = {"t0": None, "made": 0}

        def producer(c):
            if not c.running:
                return
            if state["t0"] is None:
                state["t0"] = time.perf_counter()
            due = int((time.perf_counter() - state["t0"]) * rate)
            while state["made"] < due:
                c.push(0, _image_bytes(), timestamp=state["made"] * 50_000)
                c.push(1, RAW_BYTES)
                state["made"] += 1

        cam_fake.producer = producer
        cam = OptomotiveCamera(vevo_camera=cam_fake)
        n0 = len(cam_fake.calls)  # calls made by __init__
        img = OptomotiveImage(cam)
        raw = OptomotiveRaw(cam, frame_bits=RAW_BYTES * 8)
        ldaq = LDAQ.Core(acquisitions=[img, raw])
        threads_before = threading.active_count()
        ldaq.run(measurement_duration=0.5, hotkeys=False, autostart=True, verbose=0)
        cam.close()

        meas = ldaq.get_measurement_dict()
        m_img = meas["OptomotiveImage"]
        m_raw = meas["OptomotiveRaw"]
        n_img = m_img["video"][0].shape[0]
        n_raw = m_raw["video"][0].shape[0]
        assert n_img > 0 and n_raw > 0
        assert m_img["video"][0].shape == (n_img, HEIGHT, WIDTH)
        assert m_raw["video"][0].shape == (n_raw, 1, RAW_BYTES)
        assert m_img["video"][0].dtype == np.uint8
        assert m_raw["video"][0].dtype == np.uint8
        assert cam_fake.calls.count("start") == 1
        assert cam_fake.calls[n0:].count("stop") == 1
        assert threading.active_count() <= threads_before
        dropped = img.dropped_frames()  # 200 fps: 50 000 ticks per frame
        assert dropped["n_frames"] >= n_img
        assert dropped["n_missing"] == 0
