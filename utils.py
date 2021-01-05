import os
import torch
import numpy as np
import random
import GPUtil

def mask_gpu(gpu_index=None):
    gpus = GPUtil.getGPUs()

    if gpu_index is None:
        mem_frees = [gpu.memoryFree for gpu in gpus]
        gpu_index = mem_frees.index(max(mem_frees))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_index].id)

def seed_everything(seed=1126):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
