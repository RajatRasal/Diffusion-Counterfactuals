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
from counterfactuals.scm.image_mechanism.diffusion import DiffusionImageMechanism
from counterfactuals.scm.image_mechanism.hvae import HVAEImageMechanism
from counterfactuals.scm.image_mechanism.vae import VAEImageMechanism
from counterfactuals.scm.numerical_mechanism.morphomnist import NumericalMorphoMNISTMechanisms
from data.mixer import datamodule_mixer
from data.morphomnist.datasets import MorphoMNIST
from models.classifier.mnist import MNISTClassifier
from models.ddpm.guided_diffusion import GuidedDiffusion
from models.hvae.hvae import ConditionalHVAE
from models.vae.vae import ConditionalVAE
from morphomnist.morphomnist.measure import measure_batch


@torch.no_grad()
def metrics(scm, test_dataloader, feat, classifier, last_batch=-1):
    comps = []
    revs = []
    acc_metric = Accuracy(task="multiclass", num_classes=10).to(classifier.device)
    cfs = []

    # Compute all metrics
    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # setup data
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}
        interv_batch = {k: v.clone()[idxs] if k == feat else v.clone() for k, v in factual_batch.items()}
        # composition
        cf_comp = scm.counterfactual(factual_batch, factual_batch)
        comp = l1_distance(cf_comp["image"], factual_batch["image"])
        comps.append(comp)
        # counterfactual
        cf = scm.counterfactual(factual_batch, {feat: interv_batch[feat]})
        cfs.append(cf)
        # digit effectiveness
        class_preds = classifier.predict_step({"image": cf["image"].clamp(-1, 1)}).argmax(dim=1)
        class_targets = cf["class"].argmax(dim=1)        
        acc_metric.update(class_preds, class_targets)
        # reversibility
        cf_rev = scm.counterfactual(cf, factual_batch) 
        rev = l1_distance(cf_rev["image"], factual_batch["image"])
        revs.append(rev)
        # end if we only want to test on a subset of data
        if batch_no == last_batch:
            break

    # Aggregate comp and rev
    comp = torch.cat(comps, dim=0).mean()
    rev = torch.cat(revs, dim=0).mean()

    def mape(gt, pred):
        ape = torch.abs((gt - pred) / gt)
        mape = torch.median(ape)  #.mean()
        return mape
    
    # Effectiveness for other attributes
    cfs_agg = defaultdict(list)
    for cf in cfs:
        for k, v in cf.items():
            cfs_agg[k].append(v)
    cfs_agg = {k: torch.cat(v, dim=0) for k, v in cfs_agg.items()}
    cf_images = (cfs_agg["image"].clamp(-1, 1) / 2 + 0.5) * 255
    with multiprocessing.Pool() as pool:
        measures = measure_batch(cf_images.cpu().numpy().squeeze(1), pool=pool)
    eff_metrics = {
        metric: mape(
            MorphoMNIST.unnormalise_metadata_helper(metric, cfs_agg[metric].cpu()),
            torch.tensor(measures[metric].to_numpy()),
        )
        for metric in ["thickness", "slant", "intensity"]
    }
    eff_metrics["class"] = acc_metric.compute()
    
    return comp, rev, eff_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["diffusion", "vae", "hvae"], default="diffusion")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args)

    # Load image model
    if args.model == "diffusion":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_ddpm_mnist/"
        model_folder = f"lightning_logs/{args.version}"
        hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        with open(hparams_yaml, "r") as f:
            hparams = yaml.safe_load(f.read())
        model = GuidedDiffusion(**hparams).cuda()
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights["state_dict_ema"])

        image_mechanism = DiffusionImageMechanism(
            model,
            args.guidance_scale if args.cfg else 0,
            args.timesteps,
            "ddim",
            "cfg" if args.cfg else "ddim",
            include_d=False,
        )
        output_dir = os.path.join(folder, model_folder, f"cfg_{args.cfg}_g_{args.guidance_scale}_t_{args.timesteps}")
    elif args.model == "vae":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_vae_mnist"
        model_folder = f"lightning_logs/{args.version}"
        hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        with open(hparams_yaml, "r") as f:
            hparams = yaml.safe_load(f.read())
        model = ConditionalVAE(**hparams).cuda()
        hparams["seed"] = args.seed
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights["state_dict_ema"])
        output_dir = os.path.join(folder, model_folder, "metrics")
        image_mechanism = VAEImageMechanism(model=model, deterministic=False)
    elif args.model == "hvae":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_hvae_mnist"
        model_folder = f"lightning_logs/{args.version}"
        hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        with open(hparams_yaml, "r") as f:
            hparams = yaml.safe_load(f.read())
        model = ConditionalHVAE(**hparams).cuda()
        hparams["seed"] = args.seed
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights["state_dict_ema"])
        output_dir = os.path.join(folder, model_folder, "metrics")
        image_mechanism = HVAEImageMechanism(model=model)
    else:
        raise NotImplementedError

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

    scm = SCM(image_mechanism, numerical_mechanisms)

    # Calculate metrics
    for feat in features_order:
        print("Intervention:", feat)
        comp, rev, eff = metrics(scm, dl, feat, classifier, args.last_batch)
        all_scores = {
            "comp": comp.item(),
            "rev": rev.item(),
            "eff": {k: v.item() for k, v in eff.items()}
        }
        print(all_scores)
        with open(os.path.join(output_dir, f"{feat}_metrics.json"), "w") as f:
            json.dump(all_scores, f)

    # return output_dir, model, classifier, scm, dl, args.last_batch


# def _main():
#     output_dir, model, classifier, scm, dl, last_batch = _main()
# 
#     cycles = 1
#     all_scores = {}
# 
#     # Composition
#     print("Composition")
#     l1_scores, lpips_scores = _composition(scm, dl, cycles, last_batch)
#     comp_l1 = l1_scores[0].mean()
#     comp_lpips = lpips_scores[0].mean()
#     all_scores["comp_l1"] = comp_l1.item()
#     all_scores["comp_lpips"] = comp_lpips.item()
# 
#     # Reversiblity
#     print("Reversiblity and Effectiveness")
#     for idx in [0, 1]:
#         print(f"feature {idx}")
#         l1_scores, lpips_scores, acc_interv, acc_other, fid = _reversibility(scm, dl, cycles, idx, classifier, last_batch)
#         rev_l1 = l1_scores[0].mean()
#         rev_lpips = lpips_scores[0].mean()
#         all_scores[f"rev_l1_{idx}"] = rev_l1.item()
#         all_scores[f"rev_lpips_{idx}"] = rev_lpips.item()
#         all_scores[f"eff_acc_interv_{idx}"] = acc_interv.item()
#         all_scores[f"eff_acc_other_{idx}"] = acc_other.item()
#         all_scores[f"real_fid_{idx}"] = fid.item()
# 
#     # Store metrics
#     print(all_scores)
#     with open(os.path.join(output_dir, "metrics.json"), "w") as f:
#         json.dump(all_scores, f)


if __name__ == "__main__":
    main()