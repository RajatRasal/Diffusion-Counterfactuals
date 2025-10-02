from typing import Dict, List, Literal, Optional

import torch

from counterfactuals.scm.image_mechanism.base import ImageMechanism
from models.hvae.hvae import ConditionalHVAE


class HVAEImageMechanism(ImageMechanism):

    def __init__(self, model: ConditionalHVAE):
        self.model = model

    def abduct(self, obs: Dict) -> Dict:
        image, cond = obs["image"], obs["metadata"].float()
        z, u = self.model.encode(image, cond)
        return {"z": z, "u": u}

    def predict(self, noise: Dict, cond: Dict) -> Dict:
        z, u = noise["z"], noise["u"]
        cond = cond["metadata"].float()
        x = self.model.decode(z, u, cond)
        return {"image": x}
