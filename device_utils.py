import torch

def to_device(tensor, device):
    if torch.cuda.is_available():
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor
