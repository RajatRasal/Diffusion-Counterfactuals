"""
The codes are modified.

Link:
    - [SimpleLoss] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/losses.py#L4-L15
"""
import torch
import torch.nn as nn


class SimpleLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_encoder = cfg['model']['network']['encoder']['use']
        self.gaussian = cfg['model']['network']['encoder']['gaussian']
        if self.gaussian and self.use_encoder:
            self.kl_weight = cfg['model']['network']['encoder']['kl_weight']

    def _kl_divergence(self, mean, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)

    def forward(self, out, noise, mean, logvar):
        """
        Args:
            outputs (torch.tensor): A tensor of predicted noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            noises (torch.tensor): A tensor of ground truth noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32

        Returns:
            loss (torch.tensor): A tensor of simple loss.
                shape = ()
                dtype = torch.float32
        """
        recon = (noise - out).square().sum(dim=(1, 2, 3)).mean(dim=0)
        if self.use_encoder:
            if self.gaussian:
                kl = self.kl_weight * self._kl_divergence(mean, logvar)
                loss = recon + kl
            else:
                kl = None
        else:
            loss = recon
            kl = None
        return loss, recon, kl
