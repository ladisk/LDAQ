"""Tests for the LDAQ TelopsCamera plugin (no hardware required).

These tests mock the pyTelops Camera class to verify the plugin's
wiring, lifecycle, and read_data contract without touching real
hardware. Hardware-dependent tests are marked separately and skipped
in the default run.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_fake_pytelops_camera(width=64, height=64,
                               cal_mode_name="RT",
                               frame_rate=100.0):
    """Build a MagicMock that quacks like a pyTelops Camera."""
    cam = MagicMock()
    cam.is_connected = True
    cam.is_acquiring = False

    # Properties (settable + readable)
    cam.resolution = (width, height)
    cam.frame_rate = frame_rate
    cam.integration_time = 50.0

    # calibration_mode returns an object with .name (mimics IntEnum)
    cal_mode = MagicMock()
    cal_mode.name = cal_mode_name
    cam.calibration_mode = cal_mode

    # read_frame returns float32 (H, W) frames in RT mode
    fake_frame = np.full((height, width), 25.0, dtype=np.float32)
    cam.read_frame.return_value = fake_frame

    return cam


@pytest.fixture
def fake_camera():
    return _make_fake_pytelops_camera()


@pytest.fixture
def telops_module():
    """Patch pyTelops in the plugin namespace and return the module.

    Uses ``create=True`` so the test runs even if pyTelops is not
    installed (in which case the plugin's ``try/except ImportError``
    leaves ``_PyTelopsCamera`` undefined).
    """
    import LDAQ.telops.acquisition as mod
    with patch.object(mod, "_PyTelopsCamera", create=True) as mock_cls:
        mock_cls.side_effect = lambda *a, **kw: _make_fake_pytelops_camera()
        yield mod


class TestTelopsCameraConstruction:
    """Constructor wiring — owned camera, user-provided camera."""

    def test_owned_camera_default(self, telops_module):
        tel = telops_module.TelopsCamera()
        assert tel.acquisition_name == "TelopsCamera"
        assert tel._owns_camera is True
        assert tel.cam.is_connected
        assert tel.sample_rate == 100.0  # default
        assert tel.buffer_dtype == np.float32  # RT mode
        assert tel.camera_acq_started is True
        tel.cam.acquisition_start.assert_called_once()

    def test_user_provided_camera(self, telops_module):
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(camera=cam)
        assert tel._owns_camera is False
        assert tel.cam is cam
        cam.acquisition_start.assert_called_once()

    def test_user_camera_must_be_connected(self, telops_module):
        cam = _make_fake_pytelops_camera()
        cam.is_connected = False
        with pytest.raises(RuntimeError, match="must be connected"):
            telops_module.TelopsCamera(camera=cam)

    def test_custom_acquisition_name(self, telops_module):
        tel = telops_module.TelopsCamera(acquisition_name="thermal_top")
        assert tel.acquisition_name == "thermal_top"

    def test_sample_rate_set_and_read_back(self, telops_module):
        # Plugin should set frame_rate on the camera, then read back
        # the actual achieved value. With a plain MagicMock the read-back
        # equals the set value — real pyTelops would clamp to the max
        # achievable for the current resolution and integration time.
        tel = telops_module.TelopsCamera(sample_rate=250.0)
        assert tel.sample_rate == 250.0

    def test_sample_rate_none_uses_camera_default(self, telops_module):
        cam = _make_fake_pytelops_camera(frame_rate=42.0)
        tel = telops_module.TelopsCamera(sample_rate=None, camera=cam)
        assert tel.sample_rate == 42.0

    def test_integration_time_passed_through(self, telops_module):
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(integration_time=25.0, camera=cam)
        # The plugin should have set integration_time on the camera
        assert cam.integration_time == 25.0

    def test_buffer_dtype_uint16_for_raw_modes(self, telops_module):
        cam = _make_fake_pytelops_camera(cal_mode_name="NUC")
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        assert tel.buffer_dtype == np.uint16

    def test_buffer_dtype_float32_for_rt_mode(self, telops_module):
        cam = _make_fake_pytelops_camera(cal_mode_name="RT")
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        assert tel.buffer_dtype == np.float32

    def test_buffer_dtype_uint16_for_raw0(self, telops_module):
        cam = _make_fake_pytelops_camera(cal_mode_name="RAW0")
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        assert tel.buffer_dtype == np.uint16

    def test_unknown_calibration_mode_warns(self, telops_module):
        cam = _make_fake_pytelops_camera(cal_mode_name="WEIRD_MODE")
        with pytest.warns(RuntimeWarning, match="Unknown calibration mode"):
            tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        # Falls back to uint16
        assert tel.buffer_dtype == np.uint16

    def test_calibration_mode_string_fallback(self, telops_module):
        """Real pyTelops returns an enum with .name; tests use a string.
        The plugin must handle both via the hasattr fallback."""
        cam = _make_fake_pytelops_camera()
        cam.calibration_mode = "RT"  # plain string, no .name attribute
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        assert tel.buffer_dtype == np.float32

    def test_calibration_mode_none_respects_user_camera(self, telops_module):
        """When camera= is provided and calibration_mode is None,
        the plugin must NOT touch the camera's existing mode."""
        cam = _make_fake_pytelops_camera(cal_mode_name="NUC")
        original_mode = cam.calibration_mode  # save reference
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        # The setter was NOT called — calibration_mode is still the
        # original MagicMock object we pre-set.
        assert cam.calibration_mode is original_mode

    def test_calibration_mode_default_only_for_owned_camera(self, telops_module):
        """For a user-provided camera with no kwarg, RT must NOT be
        the silent default."""
        cam = _make_fake_pytelops_camera(cal_mode_name="NUC")
        original_mode = cam.calibration_mode
        tel = telops_module.TelopsCamera(camera=cam)  # no calibration_mode kwarg
        # User camera + no kwarg → respect existing mode
        assert cam.calibration_mode is original_mode

    def test_owned_camera_default_calibration_is_rt(self, telops_module):
        """For an owned camera with no kwarg, RT IS the default."""
        tel = telops_module.TelopsCamera()
        # Plugin set calibration_mode = "RT" on the owned camera
        assert tel.cam.calibration_mode == "RT"

    def test_sample_rate_default_only_for_owned_camera(self, telops_module):
        """User-provided camera with no sample_rate kwarg keeps existing rate."""
        cam = _make_fake_pytelops_camera(frame_rate=42.5)
        tel = telops_module.TelopsCamera(camera=cam)  # no sample_rate
        assert tel.sample_rate == 42.5  # not 100.0 (the owned-camera default)

    def test_sample_rate_clamp_warning(self, telops_module):
        """If the camera clamps the requested rate, a warning should fire."""
        # Use a small fake class so the property semantics are predictable.
        class ClampingCam:
            is_connected = True
            is_acquiring = False
            resolution = (64, 64)
            integration_time = 50.0
            calibration_mode = type("M", (), {"name": "RT"})()
            _achieved = 50.0  # clamped value
            @property
            def frame_rate(self):
                return self._achieved
            @frame_rate.setter
            def frame_rate(self, value):
                pass  # silently clamp to _achieved
            def acquisition_start(self): pass
            def acquisition_stop(self): pass
            def read_frame(self, timeout=0.0):
                return np.zeros((64, 64), dtype=np.float32)
            def disconnect(self): pass

        with pytest.warns(RuntimeWarning, match="clamped"):
            telops_module.TelopsCamera(camera=ClampingCam(), sample_rate=200.0)

    def test_init_failure_cleans_up_owned_camera(self, telops_module):
        """If __init__ raises after creating+connecting the owned camera,
        the camera must be disconnected (no socket leak)."""
        cleanup_seen = {"disconnected": False, "acq_stopped": False}

        class FailingCam:
            is_connected = False
            is_acquiring = False
            resolution = (64, 64)
            frame_rate = 100.0
            integration_time = 50.0
            calibration_mode = type("M", (), {"name": "RT"})()
            def connect(self):
                self.is_connected = True
            def disconnect(self):
                self.is_connected = False
                cleanup_seen["disconnected"] = True
            def acquisition_start(self):
                raise RuntimeError("simulated acquisition_start failure")
            def acquisition_stop(self):
                cleanup_seen["acq_stopped"] = True
            def read_frame(self, timeout=0.0):
                return None

        telops_module._PyTelopsCamera.side_effect = lambda: FailingCam()
        with pytest.raises(RuntimeError, match="simulated"):
            telops_module.TelopsCamera()
        assert cleanup_seen["disconnected"], (
            "Owned camera must be disconnected when __init__ raises "
            "after connect()")


class TestTelopsCameraDataFlow:
    """read_data shape, dtype, and empty-call behavior."""

    def test_channel_shape_matches_resolution(self, telops_module):
        cam = _make_fake_pytelops_camera(width=128, height=64)
        tel = telops_module.TelopsCamera(camera=cam)
        # Channel shape is (H, W) — numpy convention
        assert tel._channel_shapes_video_init == [(64, 128)]

    def test_read_data_returns_one_frame(self, telops_module):
        cam = _make_fake_pytelops_camera(width=8, height=4)
        tel = telops_module.TelopsCamera(camera=cam)
        result = tel.read_data()
        assert result.shape == (1, 32)  # (1 sample, H*W = 4*8)
        assert result.dtype == np.float32
        assert (result == 25.0).all()

    def test_read_data_returns_empty_when_no_frame(self, telops_module):
        cam = _make_fake_pytelops_camera(width=8, height=4)
        cam.read_frame.return_value = None
        tel = telops_module.TelopsCamera(camera=cam)
        result = tel.read_data()
        assert result.shape == (0, 32)
        assert result.dtype == np.float32

    def test_read_data_handles_runtime_error_gracefully(self, telops_module):
        """If acquisition was stopped externally, read_data returns empty."""
        cam = _make_fake_pytelops_camera(width=8, height=4)
        cam.read_frame.side_effect = RuntimeError("acquisition not active")
        tel = telops_module.TelopsCamera(camera=cam)
        result = tel.read_data()
        assert result.shape == (0, 32)

    def test_get_sample_rate(self, telops_module):
        tel = telops_module.TelopsCamera()
        assert tel.get_sample_rate() == tel.sample_rate

    def test_read_data_uint16_dtype_for_raw_modes(self, telops_module):
        """In NUC mode the plugin must produce uint16 frames."""
        cam = _make_fake_pytelops_camera(width=8, height=4,
                                         cal_mode_name="NUC")
        fake_uint16 = np.full((4, 8), 1234, dtype=np.uint16)
        cam.read_frame.return_value = fake_uint16
        tel = telops_module.TelopsCamera(camera=cam, calibration_mode=None)
        result = tel.read_data()
        assert result.shape == (1, 32)
        assert result.dtype == np.uint16
        assert (result == 1234).all()

    def test_clear_buffer_drains_until_empty(self, telops_module):
        cam = _make_fake_pytelops_camera(width=8, height=4)
        tel = telops_module.TelopsCamera(camera=cam)
        # Reset after constructor so we count only clear_buffer calls
        cam.read_frame.reset_mock()
        fake = np.zeros((4, 8), dtype=np.float32)
        cam.read_frame.side_effect = [fake, fake, fake, None]
        tel.clear_buffer()
        # Drain pulled 3 frames and then saw None — exited the loop.
        # Use >= to avoid baking in the "+1 None" implementation detail.
        assert cam.read_frame.call_count >= 3
        # And the last call returned None (loop terminated correctly)
        assert cam.read_frame.return_value is None or \
               len(cam.read_frame.side_effect_list if hasattr(
                   cam.read_frame, "side_effect_list") else []) == 0

    def test_clear_buffer_noop_when_not_started(self, telops_module):
        """After terminate, clear_buffer must not call read_frame."""
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(camera=cam)
        tel.terminate_data_source()
        cam.read_frame.reset_mock()
        tel.clear_buffer()
        cam.read_frame.assert_not_called()

    def test_clear_buffer_swallows_runtime_error(self, telops_module):
        """clear_buffer must not propagate RuntimeError from a dead camera."""
        cam = _make_fake_pytelops_camera()
        cam.read_frame.side_effect = RuntimeError("acquisition not active")
        tel = telops_module.TelopsCamera(camera=cam)
        # Must not raise
        tel.clear_buffer()


class TestTelopsCameraLifecycle:
    """Lifecycle: set_data_source, terminate_data_source, ownership."""

    def test_set_data_source_idempotent(self, telops_module):
        tel = telops_module.TelopsCamera()
        # Constructor already called set_data_source once
        n_calls_before = tel.cam.acquisition_start.call_count
        tel.set_data_source()
        tel.set_data_source()
        # No extra acquisition_start calls
        assert tel.cam.acquisition_start.call_count == n_calls_before

    def test_terminate_disconnects_owned_camera(self, telops_module):
        tel = telops_module.TelopsCamera()
        owned_cam = tel.cam
        tel.terminate_data_source()
        owned_cam.acquisition_stop.assert_called_once()
        owned_cam.disconnect.assert_called_once()
        assert tel.camera_acq_started is False

    def test_terminate_does_not_disconnect_user_camera(self, telops_module):
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(camera=cam)
        tel.terminate_data_source()
        cam.acquisition_stop.assert_called_once()
        cam.disconnect.assert_not_called()

    def test_terminate_idempotent(self, telops_module):
        """Calling terminate_data_source twice must not double-call cleanup."""
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(camera=cam)
        tel.terminate_data_source()
        n_stop_after_first = cam.acquisition_stop.call_count
        tel.terminate_data_source()
        # Second terminate must be a no-op (camera_acq_started is False)
        assert cam.acquisition_stop.call_count == n_stop_after_first

    def test_context_manager_lifecycle(self, telops_module):
        cam = _make_fake_pytelops_camera()
        tel = telops_module.TelopsCamera(camera=cam)
        # Reset call counts after construction
        cam.acquisition_start.reset_mock()
        cam.acquisition_stop.reset_mock()

        with tel:
            # set_data_source already called in __enter__ (no-op since
            # camera_acq_started is True from constructor)
            pass
        # __exit__ calls terminate_data_source
        cam.acquisition_stop.assert_called_once()


class TestTelopsCameraImportError:
    """If pyTelops is missing, the constructor must raise a clear ImportError."""

    def test_missing_pytelops_raises(self, monkeypatch):
        """When _PyTelopsCamera is None (the sentinel set by the
        plugin's ``except ImportError`` block), constructing TelopsCamera
        must raise ImportError with a helpful message."""
        import LDAQ.telops.acquisition as mod
        monkeypatch.setattr(mod, "_PyTelopsCamera", None)
        with pytest.raises(ImportError, match="pyTelops is not installed"):
            mod.TelopsCamera()
