"""
Multi-GPU support for FLUX model.
Distributes model layers across available GPUs based on parameter size.
"""

import torch
import torch.nn as nn


def _param_bytes(module):
    """Calculate total parameter bytes for a module."""
    return sum(p.numel() * p.element_size() for p in module.parameters())


def distribute_model(model, num_gpus=None):
    """
    Distribute FLUX model blocks across multiple GPUs.
    Uses greedy bin-packing based on actual parameter size.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        model.to("cuda:0")
        return model

    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    # Measure all block sizes
    all_blocks = list(model.double_blocks) + list(model.single_blocks)
    block_sizes = [_param_bytes(b) for b in all_blocks]
    total_block_bytes = sum(block_sizes)

    # Embeddings + final_layer size
    embed_modules = [model.img_in, model.time_in, model.vector_in,
                     model.guidance_in, model.txt_in, model.pe_embedder,
                     model.final_layer]
    embed_bytes = sum(_param_bytes(m) for m in embed_modules)

    print(f"  Total blocks: {len(all_blocks)} ({total_block_bytes / 1e9:.1f} GB)")
    print(f"  Embeddings/final: {embed_bytes / 1e6:.0f} MB")

    # Greedy assignment: assign blocks sequentially, track load per GPU
    # Leave headroom on GPU that gets embeddings for activation memory
    gpu_load = [0] * num_gpus

    # Put embeddings on the GPU that will have least blocks (last one)
    embed_gpu = num_gpus - 1
    for m in embed_modules:
        m.to(devices[embed_gpu])
    gpu_load[embed_gpu] += embed_bytes

    # Target load per GPU (accounting for embeddings)
    target_per_gpu = (total_block_bytes + embed_bytes) / num_gpus

    block_devices = []
    gpu_id = 0
    for i, (block, size) in enumerate(zip(all_blocks, block_sizes)):
        # Move to next GPU if current one exceeds target (but not last GPU)
        if gpu_id < num_gpus - 1 and gpu_load[gpu_id] + size > target_per_gpu:
            gpu_id += 1
        block.to(devices[gpu_id])
        block_devices.append(devices[gpu_id])
        gpu_load[gpu_id] += size

    for i in range(num_gpus):
        print(f"  GPU {i}: {gpu_load[i] / 1e9:.1f} GB")

    model._block_devices = block_devices
    model._num_double = len(model.double_blocks)
    model._multi_gpu = True
    model._primary_device = devices[embed_gpu]

    return model
