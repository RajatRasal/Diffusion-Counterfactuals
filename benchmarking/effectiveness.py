from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import torch

from counterfactuals.scm.full_scm import SCM


class Oracles(ABC):

    # def __init__(self, measurement_functions: Dict[str, Callable], comparison_functions: Dict[str, Callable]):
    #     self.measurement_functions = measurement_functions
    #     self.comparison_functions = comparison_functions

    @abstractmethod
    def measure(self, images: torch.FloatTensor):
        raise NotImplementedError

    @abstractmethod
    def compare(
        self,
        image_measurements: Dict[str, torch.FloatTensor],
        gt_measurements: Dict[str, torch.FloatTensor],
    ) -> Dict[str, torch.FloatTensor]:
        raise NotImplementedError


def effectiveness(scm: SCM, factual_batch: Dict, intervs: Dict, oracles: Oracles) -> Dict[str, torch.FloatTensor], Dict[str, torch.FloatTensor]:
    cf_batch = scm.counterfactual(factual_batch, intervs)
    cf_image = cf_batch["image"]
    cf_numerical = {k: cf for k, cf in cf_batch.items() if k != "image"}
    cf_image_measurements = oracles.measure(cf_image)
    diffs = oracles.compare(cf_image_measurements, cf_numerical)
    return diffs
