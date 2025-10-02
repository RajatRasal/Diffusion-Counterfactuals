import torch


def unnormalise_image(img: torch.FloatTensor) -> torch.FloatTensor:
    """From [-1, 1] to [0, 1]"""
    return img / 2 + 0.5