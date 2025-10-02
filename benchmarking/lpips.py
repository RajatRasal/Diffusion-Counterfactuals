from typing import Literal

import torch
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update


def lpips(
    src: torch.FloatTensor,
    target: torch.FloatTensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
) -> torch.FloatTensor:
    net = _NoTrainLpips(net=net_type).to(device=src.device, dtype=src.dtype)
    loss, _ = _lpips_update(src, target, net, normalize=False)
    if len(loss.shape) == 0:
        loss = loss.unsqueeze(0)
    return loss
