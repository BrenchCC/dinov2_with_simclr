import os
import multiprocessing
import torch

def run_on_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    while True:
        with torch.inference_mode():
            a = torch.randn(200, 200, device = gpu_id)
            b = torch.randn(200, 200, device = gpu_id)
            c = torch.matmul(a, b)
            d = torch.matmul(a, c)
            e = torch.matmul(d, b)
            print(f"GPU: {gpu_id} result: {e}")

if __name__ == "__main__":
    num_gpus = min(4, torch.cuda.device_count())
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run_on_gpu, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()