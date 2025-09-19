from __future__ import annotations
import gc
import torch
import torch.distributed as dist
from .logging import configure_logging
import time


from dataclasses import dataclass
from typing import Optional
import pynvml as nvml


@dataclass
class GPUMemory:
    index: int
    name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


def get_gpu_memory(dev: int) -> GPUMemory:
    nvml.nvmlInit()
    try:
        h = nvml.nvmlDeviceGetHandleByIndex(dev)
        name: str = nvml.nvmlDeviceGetName(h)
        mem = nvml.nvmlDeviceGetMemoryInfo(h)  # <-- the call
        return GPUMemory(
            index=dev,
            name=name,
            total_bytes=mem.total,
            used_bytes=mem.used,
            free_bytes=mem.free,
        )
    finally:
        nvml.nvmlShutdown()


def pretty_sizes(bytes_val: int) -> str:
    return f"{bytes_val / 1024 / 1024 / 1024:.2f} GiB"


logger = configure_logging(__name__)


def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def available_memory(device: int = 0):
    torch.cuda.synchronize(device)
    mem = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    m = get_gpu_memory(device)
    return {
        "gpu": str(torch.cuda.current_device()),
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
        "total_GB_nvml": round(m.total_bytes / (1024**3), 2),
        "free_GB_nvml": round(m.free_bytes / (1024**3), 2),
        "used_GB_nvml": round(m.used_bytes / (1024**3), 2),
        "total_GB": round(total / (1024**3), 2),
        "free_GB": round(free / (1024**3), 2),
        "used_GB": round((total - free) / (1024**3), 2),
    }


def print_memory(msg):
    # if dist.get_rank() == 0:
    # Print utc time
    print(
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} Memory-Usage {msg} {dist.get_rank()}: {available_memory(device=dist.get_rank())}"
    )
    # logger.info(f"Memory-Usage {msg} {dist.get_rank()}:", available_memory())
