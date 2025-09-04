import pynvml
import torch


def get_gpu_usage(device: torch.device | str | int) -> float:
    pynvml.nvmlInit()

    # Normalize device input into an integer index
    if isinstance(device, torch.device):
        index = device.index if device.index is not None else 0
    elif isinstance(device, str):
        if device.startswith("cuda:"):
            index = int(device.split(":")[1])
        else:
            index = int(device)
    elif isinstance(device, int):
        index = device
    else:
        raise ValueError(f"Unsupported device type: {type(device)}")

    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    usage = mem.used / mem.total

    pynvml.nvmlShutdown()
    return usage
