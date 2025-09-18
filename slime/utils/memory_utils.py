import gc
import torch
import torch.distributed as dist
from logging import getLogger

logger = getLogger(__name__)


def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def available_memory():
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    mem = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "gpu": str(torch.cuda.current_device()),
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
        "total_GB": round(total / (1024**3), 2),
        "free_GB": round(free / (1024**3), 2),
        "used_GB": round((total - free) / (1024**3), 2),
    }


def print_memory(msg):
    # if dist.get_rank() == 0:
    logger.info(f"Memory-Usage {msg} {dist.get_rank()}:", available_memory())
