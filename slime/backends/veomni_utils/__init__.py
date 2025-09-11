import logging

import torch

try:
    from torch_memory_saver import torch_memory_saver

    _TORCH_MEMORY_SAVER_AVAILABLE = True
except ImportError:
    logging.warning("torch_memory_saver is not installed, refer to : https://github.com/fzyzcjy/torch_memory_saver")
    _TORCH_MEMORY_SAVER_AVAILABLE = False

try:
    import veomni

    _VEOMNI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VeOmni backend dependencies not available: {e}")
    _VEOMNI_AVAILABLE = False

if _VEOMNI_AVAILABLE:
    from .actor import VeOmniTrainRayActor
    from .arguments import VeOmnniFullArgs
else:

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "VeOmni backend is not available. "
            "Please ensure PyTorch with VeOmni support is installed. "
            "For installation instructions, refer to: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html"
        )

    VeOmniTrainRayActor = _raise_import_error
    VeOmnniFullArgs = _raise_import_error

__all__ = ["VeOmnniFullArgs", "VeOmniTrainRayActor"]

logging.getLogger().setLevel(logging.WARNING)
