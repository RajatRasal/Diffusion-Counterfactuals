from typing import Dict, List, Tuple

import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips_score

from counterfactuals.scm.full_scm import SCM
from counterfactuals.benchmarking.l1_distance import l1_distance
from counterfactuals.benchmarking.lpips import lpips as lpips_score
from counterfactuals.benchmarking.utils import unnormalise_image


# @torch.no_grad
def reversibility(
    scm: SCM,
    factual_batch: Dict,
    intervs: Dict,
    cycles: int = 10,
    lpips: bool = False,
) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor], List[Dict]]:
    images = [factual_batch["image"]]
    l1_scores = []
    lpips_scores = []
    for i in range(cycles):
        cf_batch = scm.counterfactual(factual_batch, intervs)
        images.append(cf_batch)
        batch_recon = scm.counterfactual(cf_batch, factual_batch)
        images.append(batch_recon)
        l1_scores.append(l1_distance(images[0], batch_recon["image"]))
        if lpips:
            lpips_scores.append(
                lpips_score(
                    unnormalise_image(images[0]),
                    unnormalise_image(batch_recon["image"]),
                    net_type="vgg"
                )
            )
        factual_batch = batch_recon
    return l1_scores, lpips_scores, images
