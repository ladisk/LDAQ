from .acquisition import NIAcquisition
from .generation import NIGeneration

try:
    from nidaqwrapper import AITask, AOTask
except ImportError:
    pass
