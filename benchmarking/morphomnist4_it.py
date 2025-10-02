import argparse
import multiprocessing
import json
import os
import random
import yaml
from collections import defaultdict
from typing import Callable, Dict

import matplotlib.pyplot as plt
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from counterfactuals.benchmarking.l1_distance import l1_distance
from counterfactuals.scm.full_scm import SCM
from counterfactuals.scm.image_mechanism.diffae import DiffusionImageMechanism
from counterfactuals.scm.image_mechanism.hvae import HVAEImageMechanism
from counterfactuals.scm.image_mechanism.vae import VAEImageMechanism
from counterfactuals.scm.numerical_mechanism.morphomnist import NumericalMorphoMNISTMechanisms
from data.mixer import datamodule_mixer
from data.morphomnist.datasets import MorphoMNIST
from models.classifier.mnist import MNISTClassifier
from models.diffae.diffae.interface import DiffusionAutoEncodersInterface
from models.hvae.hvae import ConditionalHVAE
from models.vae.vae import ConditionalVAE
from morphomnist.morphomnist.transforms import SetThickness, ImageMorphology
from morphomnist.morphomnist.measure import get_intensity, measure_batch
import numpy as np
import torch.nn.functional as F


def create_true_cf(image: np.ndarray, t: float, i: float) -> np.ndarray:
    # Set thickness
    morph = ImageMorphology(image, scale=16)
    tmp_img = morph.downscale(np.float32(SetThickness(t)(morph)))
    # Set intensity
    avg_intensity = get_intensity(tmp_img)
    # print(i, avg_intensity)
    mult = i / avg_intensity
    tmp_img = np.clip(tmp_img * mult, 0, 255)
    return tmp_img


@torch.no_grad()
def metrics(scm, test_dataloader, feat, seed, last_batch=-1):
    comps = []
    revs = []
    cfs = []
    cf_errors = []

    random.seed(seed)

    # Compute all metrics
    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        print("Batch no:", batch_no)

        # setup data
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}

        # composition
        print("Composition")
        noise = scm.abduct(factual_batch)
        cf_comp = scm.predict(noise, factual_batch)
        comp = l1_distance(cf_comp["image"].clamp(-1, 1), factual_batch["image"])
        comps.append(comp)

        # counterfactual
        print("Counterfactual")
        if feat == "thickness":
            interv_batch = {"metadata": factual_batch["metadata"].clone()}
            interv_batch["metadata"] = interv_batch["metadata"][idxs]
        elif feat == "intensity":
            interv_batch = {"metadata": factual_batch["metadata"].clone()}
            interv_batch["metadata"][:, 1] = interv_batch["metadata"][idxs, 1]
        cf = scm.predict(noise, interv_batch)
        cfs.append({**cf, **interv_batch})

        # compare to true cf
        true_cf = torch.cat([
            torch.tensor(
                create_true_cf(
                    (img.cpu().numpy()[0] * 0.5 + 0.5) * 255,
                    MorphoMNIST.unnormalise_metadata_helper("thickness", interv_batch["metadata"][idx, 0]).cpu().numpy(),
                    MorphoMNIST.unnormalise_metadata_helper("intensity", interv_batch["metadata"][idx, 1]).cpu().numpy(),
                )
            ).unsqueeze(0).unsqueeze(0)
            for idx, img in enumerate(factual_batch["image"])
        ], dim=0).cuda()
        true_cf = ((true_cf / 255) - 0.5) * 2
        # cf_error = F.mse_loss(true_cf, cf["image"], reduction="none")
        cf_error = l1_distance(cf["image"].clamp(-1, 1) * 0.5 + 0.5, true_cf * 0.5 + 0.5)  # , reduction="none")
        # cf_error = F.mse_loss(cf["image"].clamp(-1, 1) * 0.5 + 0.5, true_cf * 0.5 + 0.5, reduction="none")
        cf_errors.append(cf_error)

        # reversibility
        print("Reversibility")
        cf_noise = scm.abduct({**cf, **interv_batch})
        cf_rev = scm.predict(cf_noise, factual_batch)
        rev = l1_distance(cf_rev["image"].clamp(-1, 1), factual_batch["image"])
        revs.append(rev)
        # end if we only want to test on a subset of data
        if batch_no == last_batch:
            break

    # Aggregate comp and rev
    # comp = torch.cat(comps, dim=0).mean()
    # revs = torch.cat(revs, dim=0).mean()
    cf_errors = torch.cat(cf_errors, dim=0).mean()

    # def mape(gt, pred):
    #     ape = torch.abs((gt - pred) / gt)
    #     mape = ape.mean()
    #     return mape
    
    # # Effectiveness for other attributes
    # cfs_agg = defaultdict(list)
    # for cf in cfs:
    #     for k, v in cf.items():
    #         cfs_agg[k].append(v)
    # cfs_agg = {k: torch.cat(v, dim=0) for k, v in cfs_agg.items()}
    # cfs_agg["thickness"] = cfs_agg["metadata"][:, 0]
    # cfs_agg["intensity"] = cfs_agg["metadata"][:, 1]
    # cf_images = (cfs_agg["image"].clamp(-1, 1) / 2 + 0.5) * 255
    # with multiprocessing.Pool() as pool:
    #     measures = measure_batch(cf_images.cpu().numpy().squeeze(1), pool=pool)
    # eff_metrics = {
    #     metric: mape(
    #         MorphoMNIST.unnormalise_metadata_helper(metric, cfs_agg[metric].cpu()),
    #         torch.tensor(measures[metric].to_numpy()),
    #     )
    #     for metric in ["thickness", "intensity"]
    # }
    
    # return comp, revs, eff_metrics, cf_errors
    return cf_errors  # comp, revs, eff_metrics, cf_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    # parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args)

    # Load image model
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_it"
    model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_it2"
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_it"
    print(model_dir)
    model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_500_ckpt.pth'}, 'test')
    _ = model.model.eval()
    hparams = {"seed": model.cfg["general"]["seed"]}
    print("---------------", args)
    guidance_scale = args.guidance_scale
    image_mechanism = DiffusionImageMechanism(model, guidance_scale=guidance_scale)
    output_dir = os.path.join(model_dir, f"prelim_sem_trial_2_{args.semantic}_cfg_{args.cfg}_g_{guidance_scale}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test dataset
    dm, _, _, _ = datamodule_mixer(
        dataset="mnist",
        batch_size=args.batch_size,
        num_workers=4,
        prefetch_factor=2,
        split_ratio=(0.9, 0.1),
        seed=hparams["seed"],
        resolution=64,
        cache=False,
        random_crop=False,
        normalise_metadata=True,
        one_hot_classes=True,
        features=["thickness", "intensity"],
    )
    dm.setup("test")
    dl = dm.test_dataloader()

    # SCM
    # features_order = ["thickness", "slant", "intensity", "class"]
    # numerical_mechanisms = NumericalMorphoMNISTMechanisms(
    #     normalised=True,
    #     metadata_features_order=features_order,
    #     device=torch.device('cuda'),
    # )
    # scm = SCM(image_mechanism, numerical_mechanisms)
    scm = image_mechanism

    # Calculate metrics
    for feat in ["thickness", "intensity"]:  # "slant", "class"]: # features_order:
        print("Intervention:", feat)
        cf_error = metrics(scm, dl, feat, args.seed, args.last_batch)
        print(cf_error.item())
        # all_scores = {
        #     "comp": comp.item(),
        #     "rev": rev.item(),  # {k: v.item() for k, v in rev.items()},
        #     "eff": {k: v.item() for k, v in eff.items()},  # {ce_name: {k: v.item() for k, v in ce.items()} for ce_name, ce in eff.items()}
        #     "cf_errors": cf_error.item()
        # }
        # print(all_scores)
        with open(os.path.join(output_dir, f"{feat}_metrics.json"), "w") as f:
            json.dump(all_scores, f)


if __name__ == "__main__":
    main()