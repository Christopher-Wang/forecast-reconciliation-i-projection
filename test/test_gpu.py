import torch, gc

import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).parent.parent.resolve())

from src.utils.gpu_utils import get_gpu_usage

def f(a, b, device="cuda:0", label="", handle_gpu: bool = True):
    print(f"[{label}] Before: GPU usage = {get_gpu_usage(device):.3f}")
    a = a.to(device)
    b = b.to(device)
    out = a @ b
    if handle_gpu:
        out_cpu = out.detach().to("cpu")
        del a, b, out
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        out_cpu = out
    print(f"[{label}] After:  GPU usage = {get_gpu_usage(device):.3f}")
    return out_cpu

def tree_computation(x_list, device="cuda:0", handle_gpu: bool = True):
    x12 = f(x_list[0], x_list[1], device, "x12", handle_gpu)
    x34 = f(x_list[2], x_list[3], device, "x34", handle_gpu)
    x56 = f(x_list[4], x_list[5], device, "x56", handle_gpu)
    x78 = f(x_list[6], x_list[7], device, "x78", handle_gpu)

    x1234 = f(x12, x34, device, "x1234", handle_gpu)
    x5678 = f(x56, x78, device, "x5678", handle_gpu)

    y = f(x1234, x5678, device, "y", handle_gpu=False)
    return y

if __name__ == "__main__":
    N = 6000
    x_list = [torch.randn(N, N, requires_grad=True) for _ in range(8)]

    for handle_gpu in [False, True]:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n--- handle_gpu = {handle_gpu} ---")
        y = tree_computation(x_list, handle_gpu=handle_gpu)
        print(f"[Backward start] GPU usage = {get_gpu_usage('cuda:0'):.3f}")
        y.sum().backward()
        print(f"[Backward end]   GPU usage = {get_gpu_usage('cuda:0'):.3f}")

        for i, x in enumerate(x_list, 1):
            assert x.grad is not None
