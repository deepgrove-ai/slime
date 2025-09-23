"""
FSDP v2 Utilities

This module provides utilities for FSDP v2 (fully_shard) support only.
FSDP v1 is no longer supported and will raise errors if detected.
"""

import torch
from torch.distributed.tensor import DTensor


def preprocess_tensor_for_update_weights(tensor):
    """
    Preprocess tensor for weight updates - FSDP v2 only (DTensor support).

    Args:
        tensor: The tensor to preprocess (DTensor or regular Tensor)

    Returns:
        torch.Tensor: Regular tensor ready for weight updates
    """
    if isinstance(tensor, DTensor):
        # FSDP v2 case - convert DTensor to full tensor
        return tensor.full_tensor()

    # Regular tensor - return as is
    return tensor
