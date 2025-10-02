import torch


def l1_distance(src: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
    return (src - target).abs().mean(dim=(1, 2, 3))
