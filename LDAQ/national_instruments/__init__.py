from .acquisition import NIAcquisition
from .generation import NIGeneration

try:
    from nidaqwrapper import (
        AITask, AOTask,
        get_connected_devices, list_devices, list_tasks, get_task_by_name,
    )
except ImportError:
    def _missing_nidaqwrapper(*args, **kwargs):
        raise ImportError(
            "nidaqwrapper is required. "
            "Install it with: pip install LDAQ[ni]"
        )
    AITask = _missing_nidaqwrapper
    AOTask = _missing_nidaqwrapper
    get_connected_devices = _missing_nidaqwrapper
    list_devices = _missing_nidaqwrapper
    list_tasks = _missing_nidaqwrapper
    get_task_by_name = _missing_nidaqwrapper
