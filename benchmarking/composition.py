from typing import Dict, List, Tuple

import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips_score

from counterfactuals.scm.full_scm import SCM
from counterfactuals.benchmarking.l1_distance import l1_distance
from counterfactuals.benchmarking.lpips import lpips as lpips_score
from counterfactuals.benchmarking.utils import unnormalise_image


# @torch.no_grad
def composition(
    scm: SCM,
    factual_batch: Dict,
    cycles: int = 10,
    lpips: bool = False,
) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor], List[torch.FloatTensor]]:
    images = [factual_batch["image"]]
    l1_scores = []
    lpips_scores = []
    for i in range(cycles):
        cf_batch = scm.counterfactual(factual_batch, factual_batch)
        images.append(cf_batch["image"])
        l1_scores.append(l1_distance(images[0], cf_batch["image"]))
        if lpips:
            lpips_scores.append(
                lpips_score(
                    unnormalise_image(images[0]),
                    unnormalise_image(cf_batch["image"]),
                    net_type="vgg"
                )
            )
        factual_batch = cf_batch
    return l1_scores, lpips_scores, images
