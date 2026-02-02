"""
Distributed training utilities for PyTorch DDP.

Supports SLURM, torchrun, and single-GPU modes.
"""
import os
import torch.distributed as dist


def setup_ddp():
    """
    Initialize DDP from environment variables.

    Supports:
        - torchrun / torch.distributed.launch (RANK, LOCAL_RANK, WORLD_SIZE)
        - SLURM (SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS)

    Returns:
        tuple: (rank, local_rank, world_size)

    Raises:
        RuntimeError: If no distributed environment detected
    """
    if "RANK" in os.environ:
        # torchrun or torch.distributed.launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        # SLURM
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
        world_size = int(os.environ["SLURM_NTASKS"])

        # Setup master address/port for SLURM
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        raise RuntimeError(
            "Cannot detect distributed environment. "
            "Use torchrun, torch.distributed.launch, or SLURM."
        )

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    return rank, local_rank, world_size


def cleanup_ddp():
    """Destroy the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0
