import torch
import torch.nn as nn

from .encoder import SemanticEncoder
from .unet import Unet


class DiffusionAutoEncoders(nn.Module):

    def __init__(self, cfg):
        """
        Args:
            cfg: A dict of config.
        """
        super().__init__()
        self.cfg = cfg
        self.use_semantic_encoder = cfg["model"]["network"]["encoder"]["use"]
        self.cond_semantic_encoder = cfg["model"]["network"]["encoder"]["cond"]
        cfg["model"]["network"]["unet"]["emb_channels"] += len(cfg["general"].get("attributes", []))
        if self.use_semantic_encoder:
            # Note: Look at _init_model in interface.py
            # Semantic Encoding
            cfg["model"]["network"]["unet"]["emb_channels"] += cfg["model"]["network"]["encoder"]["emb_channels"]
            data = self.cfg["general"]["data_name"]
            if self.cond_semantic_encoder:
                maps = {"mnist": 13, "celebahq": 7, "embed": 4}
                cfg["model"]["network"]["encoder"]["cond_dim"] = maps[data]
            self.unet = Unet(cfg)
            self.encoder = SemanticEncoder(cfg)
            self.gaussian_encoder = cfg['model']['network']['encoder']['gaussian']
        else:
            # DDIM
            self.unet = Unet(cfg)
        self.emb_dim = cfg["model"]["network"]["unet"]["emb_channels"]
        # CFG
        self.p_uncond = cfg['model']['p_uncond']
        if self.p_uncond >= 0:
            self.null_token = nn.Parameter(torch.randn(1, self.emb_dim))

    def reparameterisation(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x0, xt, t, y=None):
        """
        Args:
            x0 (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            xt (torch.tensor): A tensor of x at time step t.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            t (torch.tensor): A tensor of time steps.
                shape = (batch, )
                dtype = torch.float32
            y (torch.tensor): A tensor of attributes
                shape = (batch, attributes)
                dtype = torch.float32

        Returns:
            out (torch.tensor): A tensor of output.
                shape = (batch, channels, height, width)
                dtype = torch.float32
        """
        if self.use_semantic_encoder:
            if self.gaussian_encoder:
                mean, logvar = self.encoder(x0, y, use_smoothing=True)
                z = self.reparameterisation(mean, logvar)
            else:
                mean, logvar = None, None
                z = self.encoder(x0, y, use_smoothing=True)
            if y is not None:
                z = torch.cat([z, y], dim=1)
        else:
            mean, logvar = None, None
            if y is not None:
                z = y
            else:
                raise NotImplementedError("Unconditional DDIM not implemented")

        if hasattr(self, 'null_token'):
            mask = torch.rand(x0.shape[0]) < self.p_uncond
            z[mask] = self.null_token.repeat(mask.sum(), 1)
        out = self.unet(xt, t, z)
        return out, z, mean, logvar
