from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.scm.base import Mechanism


class CelebANumericalMechanisms(Mechanism):

    def __init__(self, metadata_features_order: List[str], device: torch.device):
        self.metadata_features_order = metadata_features_order
        self.device = device

    def _mouth_open_transform(self, o, s):
        return o ^ s  # xor

    def abduct(self, cond: Dict) -> Dict:
        s = cond["Smiling"]
        o = cond["Mouth_Slightly_Open"]
        return {
            "eps_s": s,  # smiling
            "eps_o": o,  # self._mouth_open_transform(o.int(), s.int()).float(),  # mouth open
            "eps_m": cond["Male"],  # male
            "eps_g": cond["Eyeglasses"],  # eyeglasses
            "eps_l": cond["Wearing_Lipstick"],  # wearing lipstick
            "eps_b": cond["Bald"],  # bald
            "eps_h": cond["Wearing_Hat"],  # wearing hat
        }

    def predict(self, noise: Dict, interv: Dict) -> Dict:
        s = interv.get("Smiling", noise["eps_s"]).to(self.device)
        preds = {
            "Smiling": s,
            "Mouth_Slightly_Open": interv.get("Mouth_Slightly_Open", s).to(self.device),
            "Male": interv.get("Male", noise["eps_m"]).to(self.device),
            "Eyeglasses": interv.get("Eyeglasses", noise["eps_g"]).to(self.device),
            "Wearing_Lipstick": interv.get("Wearing_Lipstick", noise["eps_l"]).to(self.device),
            "Bald": interv.get("Bald", noise["eps_b"]).to(self.device),
            "Wearing_Hat": interv.get("Wearing_Hat", noise["eps_h"]).to(self.device),
        }
        metadata = [preds[f].unsqueeze(1) for f in self.metadata_features_order]
        preds["metadata"] = torch.cat(metadata, dim=1)
        return preds
