from typing import Dict, List, Literal, Optional

import torch

from counterfactuals.scm.image_mechanism.base import ImageMechanism
from models.vae.vae import ConditionalVAE


class VCIImageMechanism(ImageMechanism):

    def __init__(self, model, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic

    def abduct(self, obs: Dict) -> Dict:
        image, cond = obs["image"], obs["metadata"].float()
        cov = [cov.repeat(len(cond), 1).to(image.device) for cov in [torch.tensor([0.])]]
        z, _ = self.model.encode(image, cond, cov)
        return {"z": z}

    def predict(self, noise: Dict, cond: Dict) -> Dict:
        z = noise["z"]
        cond = cond["metadata"].float()
        x, _ = self.model.decode([d.mean for d in z], cond)
        if self.deterministic:
            return {"image": x.mean, "metadata": cond}
        else:
            return {"image": x, "metadata": cond}
