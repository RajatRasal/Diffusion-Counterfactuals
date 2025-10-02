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
from data.morphomnist.datasets import MorphoMNIST


class NumericalMorphoMNISTMechanisms(Mechanism):

    def __init__(self, normalised: bool, metadata_features_order: List[str], device: torch.device):
        self.normalised = normalised
        self.metadata_features_order = metadata_features_order
        self.device = device

    def _intensity_transform(self, t, inv):
        transform = ComposeTransform([
            AffineTransform(2 * t - 5, 0.5),
            SigmoidTransform(),
            AffineTransform(64., 191.),
        ])
        return transform.inv if inv else transform

    def _thickness_transform(self):
        return AffineTransform(0.5, 1)

    def _slant_transform(self, d):
        return AffineTransform(6 * d - 27, 3)

    def abduct(self, cond: Dict) -> Dict:
        _class = cond["class"].argmax(dim=1)
        if self.normalised:
            cond = {
                k: MorphoMNIST.unnormalise_metadata_helper(k, v) if k in ["intensity", "thickness", "slant"] else v
                for k, v in cond.items() 
            }
        return {
            "eps_t": self._thickness_transform()._inverse(cond["thickness"]).to(self.device),
            "eps_i": self._intensity_transform(cond["thickness"], True)(cond["intensity"]).to(self.device),
            "eps_s": torch.tensor([self._slant_transform(c.item())._inverse(np.rad2deg(s.item())) for s, c in zip(cond["slant"], _class)]).to(self.device),
            "class": cond["class"],
        }

    def predict(self, noise: Dict, interv: Dict) -> Dict:
        if self.normalised:
            interv = {
                k: MorphoMNIST.unnormalise_metadata_helper(k, v) if k in ["intensity", "thickness", "slant"] else v
                for k, v in interv.items() 
            }
        t = interv.get("thickness", self._thickness_transform()(noise["eps_t"])).to(self.device)
        i = interv.get("intensity", self._intensity_transform(t, False)(noise["eps_i"])).to(self.device)
        d = interv.get("class", noise["class"]).to(self.device)
        _class = d.argmax(dim=1)
        s = interv.get("slant", torch.tensor([np.deg2rad(self._slant_transform(c.item())(eps_s.item())) for c, eps_s in zip(_class, noise["eps_s"])])).to(self.device)
        s = s.clamp(-0.5, 0.5)
        preds = {
            "intensity": MorphoMNIST.normalise_metadata_helper("intensity", i) if self.normalised else i,
            "thickness": MorphoMNIST.normalise_metadata_helper("thickness", t) if self.normalised else t,
            "slant": MorphoMNIST.normalise_metadata_helper("slant", s) if self.normalised else s,
            "class": d,
        }
        metadata = [preds[f].unsqueeze(1) if f != "class" else preds[f] for f in self.metadata_features_order]
        preds["metadata"] = torch.cat(metadata, dim=1)
        return preds

    def concat_metadata(self, metadata) -> Dict:
        preds = {
            "intensity": metadata["intensity"],
            "thickness": metadata["thickness"],
            "slant": metadata["slant"],
            "class": metadata["class"],
        }
        metadata = [preds[f].unsqueeze(1) if f != "class" else preds[f] for f in self.metadata_features_order]
        preds["metadata"] = torch.cat(metadata, dim=1)
        return preds


class NumericalColourMorphoMNISTMechanisms(Mechanism):

    def __init__(self, normalised: bool, metadata_features_order: List[str], device: torch.device):
        self.normalised = normalised
        self.metadata_features_order = metadata_features_order
        self.device = device

    def _thickness_transform(self):
        return AffineTransform(0.5, 1)

    def _slant_transform(self, d):
        return AffineTransform(6 * d - 27, 3)

    def _hue_transform(self, d):
        return AffineTransform(d / 10 + 0.05, 0.05)

    def abduct(self, cond: Dict) -> Dict:
        _class = cond["class"].argmax(dim=1)
        if self.normalised:
            cond = {
                k: MorphoMNIST.unnormalise_metadata_helper(k, v) if k in ["thickness", "slant"] else v
                for k, v in cond.items() 
            }
        return {
            "eps_t": self._thickness_transform()._inverse(cond["thickness"]).to(self.device),
            "eps_h": torch.tensor([self._hue_transform(c.item())._inverse(h.item()) for h, c in zip(cond["hue"], _class)]).to(self.device),
            "eps_s": torch.tensor([self._slant_transform(c.item())._inverse(np.rad2deg(s.item())) for s, c in zip(cond["slant"], _class)]).to(self.device),
            "class": cond["class"],
        }

    def predict(self, noise: Dict, interv: Dict) -> Dict:
        if self.normalised:
            interv = {
                k: MorphoMNIST.unnormalise_metadata_helper(k, v) if k in ["thickness", "slant"] else v
                for k, v in interv.items() 
            }
        t = interv.get("thickness", self._thickness_transform()(noise["eps_t"])).to(self.device)
        d = interv.get("class", noise["class"]).to(self.device)
        _class = d.argmax(dim=1)
        h = interv.get("hue", torch.tensor([self._hue_transform(c.item())(eps_h.item()) for c, eps_h in zip(_class, noise["eps_h"])])).to(self.device)
        s = interv.get("slant", torch.tensor([np.deg2rad(self._slant_transform(c.item())(eps_s.item())) for c, eps_s in zip(_class, noise["eps_s"])])).to(self.device)
        s = s.clamp(-0.5, 0.5)
        preds = {
            "thickness": MorphoMNIST.normalise_metadata_helper("thickness", t) if self.normalised else t,
            "slant": MorphoMNIST.normalise_metadata_helper("slant", s) if self.normalised else s,
            "hue": h,
            "class": d,
        }
        metadata = [
            preds[f].unsqueeze(1) if f != "class" else preds[f]
            for f in self.metadata_features_order
        ]
        preds["metadata"] = torch.cat(metadata, dim=1)
        return preds

    def concat_metadata(self, metadata) -> Dict:
        preds = {
            "thickness": metadata["thickness"],
            "slant": metadata["slant"],
            "hue": metadata["hue"],
            "class": metadata["class"],
        }
        metadata = [preds[f].unsqueeze(1) if f != "class" else preds[f] for f in self.metadata_features_order]
        preds["metadata"] = torch.cat(metadata, dim=1)
        return preds
