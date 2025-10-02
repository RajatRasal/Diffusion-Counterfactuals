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

from counterfactuals.benchmarking.composition import composition
from counterfactuals.benchmarking.reversibility import reversibility
from counterfactuals.scm.full_scm import SCM
from counterfactuals.scm.image_mechanism.diffusion import DiffusionImageMechanism
from counterfactuals.scm.numerical_mechanism.morphomnist import NumericalMorphoMNISTMechanisms
from data.mixer import datamodule_mixer
from data.morphomnist.datasets import MorphoMNIST
from models.classifier.mnist import MNISTClassifier
from models.ddpm.guided_diffusion import GuidedDiffusion
from morphomnist.morphomnist.measure import measure_batch


def _composition(scm, test_dataloader, cycles, last_batch=-1):
    l1_scores = []
    lpips_scores = []
    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch = {k: v.cuda() for k, v in batch.items()}
        l1, _, _ = composition(scm, batch, cycles, False)
        l1_scores.append(torch.cat([x.unsqueeze(0) for x in l1], dim=0))
        if idx == last_batch:
            break
    l1_scores = torch.cat(l1_scores, dim=1)
    return l1_scores[0].mean(), l1_scores[-1].mean()


def _reversibility(scm, test_dataloader, cycles, interv_feature, classifier, seed, last_batch=-1):
    random.seed(seed)

    l1_scores_first_cycle = []
    l1_scores_final_cycle = []
    cfs = []
    acc_metric = Accuracy(task="multiclass", num_classes=10).to(classifier.device)

    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # shuffle batch idxs
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}
        interv_batch = {k: v[idxs].cuda().clone() for k, v in batch.items() if k == interv_feature}
        # calculate reversibility
        l1, _, images = reversibility(scm, factual_batch, interv_batch, cycles, lpips=False)
        l1_scores_first_cycle.append(l1[2].unsqueeze(0))
        l1_scores_final_cycle.append(l1[-1].unsqueeze(0))
        # store cfs
        cf = images[1]
        cfs.append(cf)
        # digit effectiveness - calculate accuracy
        class_preds = classifier.predict_step(cf).argmax(dim=1)
        acc_metric.update(class_preds, cf["class"].argmax(dim=1))
        if batch_no == last_batch:
            break

    # Aggregate accuracy and reversiblity
    acc = acc_metric.compute()
    l1_scores_first = torch.cat(l1_scores_first_cycle, dim=1)
    l1_scores_final = torch.cat(l1_scores_final_cycle, dim=1)

    # Effectiveness metrics
    cfs_agg = defaultdict(list)
    for cf in cfs:
        for k, v in cf.items():
            cfs_agg[k].append(v)
    cfs_agg = {k: torch.cat(v, dim=0) for k, v in cfs_agg.items()}
    cf_images_agg = (cfs_agg["image"] / 2 + 0.5) * 255
    with multiprocessing.Pool() as pool:
        measures = measure_batch(cf_images_agg.cpu().numpy().squeeze(1), pool=pool)
    eff_metrics = {
        metric: torch.nn.functional.l1_loss(
            MorphoMNIST.unnormalise_metadata_helper(metric, cfs_agg[metric].cpu()),
            torch.tensor(measures[metric].to_numpy())
        )
        for metric in ["thickness", "slant", "intensity"]
    }

    return acc, l1_scores_first.mean(), l1_scores_final.mean(), eff_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_size = 1024

    # Load image model
    folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_ddpm_mnist/"
    model_folder = f"lightning_logs/{args.version}"
    hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
    with open(hparams_yaml, "r") as f:
        hparams = yaml.safe_load(f.read())
    model = GuidedDiffusion(**hparams).cuda()
    ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
    weights = torch.load(ckpt_path)
    model.load_state_dict(weights["state_dict_ema"])

    # Create output directory
    output_dir = os.path.join(
        folder,
        model_folder,
        f"g_{args.guidance_scale}_t_{args.timesteps}_s_{args.seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Test dataset
    dm, _, _ = datamodule_mixer(
        dataset="mnist",
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=None,
        split_ratio=(0.9, 0.1),
        seed=hparams["seed"],
        resolution=64,
        cache=False,
        random_crop=False,
        normalise_metadata=True,
        one_hot_classes=True,
    )
    dm.setup("test")
    dl = dm.test_dataloader()

    # Classifier
    classifier = MNISTClassifier.load_from_checkpoint(
        "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_classifier_mnist/lightning_logs/version_0/checkpoints/last.ckpt",
    )

    # SCM
    features_order = ["thickness", "slant", "intensity", "class"]
    numerical_mechanisms = NumericalMorphoMNISTMechanisms(
        normalised=True,
        metadata_features_order=features_order,
        device=torch.device('cuda'),
    )
    image_mechanism = DiffusionImageMechanism(
        model,
        args.guidance_scale,
        args.timesteps,
        "ddim",
        "cfg",
        include_d=False,
    )
    scm = SCM(image_mechanism, numerical_mechanisms)

    cycles = 5
    all_scores = {}

    # Composition
    print("Composition")
    comp_first, comp_last = _composition(scm, dl, cycles, args.last_batch)
    all_scores["comp_first"] = comp_first.item()
    all_scores["comp_last"] = comp_last.item()
    print(f"comp first: {comp_first.item()}, comp last: {comp_last.item()}")

    # Reversiblity
    print("Reversiblity and Effectiveness")
    for feature in ["thickness", "intensity", "slant", "class"]:
        print(f"feature {feature}")
        acc, rev_first, rev_final, eff_metrics = _reversibility(scm, dl, cycles, feature, classifier, args.seed, args.last_batch)
        all_scores[f"{feature}/eff_class"] = acc.item()
        all_scores[f"{feature}/rev_first"] = rev_first.item()
        all_scores[f"{feature}/rev_final"] = rev_final.item()
        for k, v in eff_metrics.items():
            all_scores[f"{feature}/eff_{k}"] = v.item()
        print({k: v for k, v in all_scores.items() if k.startswith(feature)})

    # Store metrics
    print(all_scores)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_scores, f)


if __name__ == "__main__":
    main()