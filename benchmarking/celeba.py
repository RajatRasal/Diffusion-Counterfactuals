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
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from counterfactuals.benchmarking.l1_distance import l1_distance
from counterfactuals.scm.full_scm import SCM
from counterfactuals.scm.image_mechanism.diffae import DiffusionImageMechanism
from counterfactuals.scm.image_mechanism.hvae import HVAEImageMechanism
from counterfactuals.scm.image_mechanism.vae import VAEImageMechanism
from counterfactuals.scm.numerical_mechanism.celeba import CelebANumericalMechanisms
from data.mixer import datamodule_mixer
from models.classifier.cnn import ResNetBinaryClassifier
from models.diffae.diffae.interface import DiffusionAutoEncodersInterface
from models.hvae.hvae import ConditionalHVAE
from models.vae.vae import ConditionalVAE


@torch.no_grad()
def metrics(scm, test_dataloader, feat, classifier_glasses, classifier_smiling, last_batch=-1):
    comps = []
    revs = []
    acc_glasses = Accuracy(task="binary").to(classifier_glasses.device)
    acc_smiling = Accuracy(task="binary").to(classifier_smiling.device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex',
        compute_on_cpu=True,
    ).to(classifier_glasses.device)
    fid_metric_cf = FrechetInceptionDistance(
        feature=64,
        normalize=True,
        compute_on_cpu=True,
        input_img_size=(3, 64, 64),
    ).to(classifier_glasses.device)

    # Compute all metrics
    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        print("batch no:", batch_no)
        # setup data
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}
        # composition
        print("composition")
        cf_comp = scm.counterfactual(factual_batch, factual_batch)
        comp = l1_distance(cf_comp["image"], factual_batch["image"])
        comps.append(comp)
        # counterfactual
        print("cf")
        cf = scm.counterfactual(factual_batch, {feat: (~factual_batch[feat].bool()).float()})
        # realism
        fid_metric_cf.update((factual_batch["image"] / 2 + 0.5) * 255, real=True)
        fid_metric_cf.update((cf["image"] / 2 + 0.5) * 255, real=False)
        # effectiveness
        cf_preds_glasses = classifier_glasses({"image": cf["image"]})
        acc_glasses.update(cf_preds_glasses.flatten(), cf["Eyeglasses"])
        cf_preds_smiling = classifier_smiling({"image": cf["image"]})
        acc_smiling.update(cf_preds_smiling.flatten(), cf["Smiling"])
        # reversibility
        print("reversibility")
        cf_rev = scm.counterfactual(cf, factual_batch)
        rev = l1_distance(cf_rev["image"], factual_batch["image"])
        revs.append(rev)
        # lpips
        lpips.update(cf_comp["image"], cf["image"])

        if batch_no == last_batch:
            break

    idx = [k for k, _ in factual_batch.items() if k not in ["image", "metadata"]].index(feat)
    acc_glasses = acc_glasses.compute()
    acc_smiling = acc_smiling.compute()
    fid_cf = fid_metric_cf.compute()
    lpips_score = lpips.compute()

    return torch.cat(comps, dim=0).mean(), torch.cat(revs, dim=0).mean(), acc_glasses, acc_smiling, fid_cf, lpips_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["diffusion", "vae", "hvae"], default="diffusion")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--val", action="store_true")
    args = parser.parse_args()

    # Load image model
    if args.model == "diffusion":
        if args.semantic:
            if args.cfg:
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/sem_enc_long_train'
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.1_pa_cond_2'
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.1'
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.2'
            else:
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/sem_enc_cond_long_train'
        else:
            if args.cfg:
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/ddim_long_train'
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.5_no_sem'
            else:
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/ddim_cond_long_train'
        model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_5900_ckpt.pth'}, 'test')
        _ = model.model.eval()
        hparams = {"seed": model.cfg["general"]["seed"]}
        print(args, model_dir)
        guidance_scale = args.guidance_scale if args.cfg else 1
        image_mechanism = DiffusionImageMechanism(model, guidance_scale=guidance_scale)
        output_dir = os.path.join(model_dir, f"cfg_{args.cfg}_g_{guidance_scale}")
    elif args.model == "hvae":
        print("HERE")
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_hvae_celebahq"
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
    elif args.model == "vae":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_vae_celebahq"
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
        image_mechanism = VAEImageMechanism(model=model)
    else:
        raise NotImplementedError

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test dataset
    dm, _, _, features = datamodule_mixer(
        dataset="celebahq",
        batch_size=args.batch_size,
        num_workers=4,
        prefetch_factor=None,
        split_ratio=(0.9, 0.1),
        seed=hparams["seed"],
        resolution=64,
        cache=False,
        **{"horizontal_flip_prob": 0}
    )
    dm.setup("fit" if args.val else "test")
    dl = dm.val_dataloader() if args.val else dm.test_dataloader()

    numerical_mechanism = CelebANumericalMechanisms(
        features,
        torch.device('cuda'),
    )

    scm = SCM(image_mechanism, numerical_mechanism)

    # Classifier
    version = "version_35"
    folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_classifier_eyeglasses_celebahq"
    model_folder = f"lightning_logs/{version}"
    hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
    with open(hparams_yaml, "r") as f:
        hparams = yaml.safe_load(f.read())
    classifier_glasses = ResNetBinaryClassifier(**hparams).cuda().float()
    ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
    weights = torch.load(ckpt_path)
    classifier_glasses.load_state_dict(weights["state_dict_ema"])

    version = "version_1"
    folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_classifier_smiling_celebahq"
    model_folder = f"lightning_logs/{version}"
    hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
    with open(hparams_yaml, "r") as f:
        hparams = yaml.safe_load(f.read())
    classifier_smiling = ResNetBinaryClassifier(**hparams).cuda().float()
    ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
    weights = torch.load(ckpt_path)
    classifier_smiling.load_state_dict(weights["state_dict_ema"])

    # Calculate metrics
    for attr in ["Smiling", "Eyeglasses"]:
        print("Intervention:", attr)
        comp, rev, acc_glasses, acc_smiling, fid_cf, lpips_score = metrics(scm, dl, attr, classifier_glasses, classifier_smiling, args.last_batch)

        all_scores = {
            "comp": comp.item(),
            "rev": rev.item(),
            "acc_g": acc_glasses.item(),
            "acc_s": acc_smiling.item(),
            "fid_cf": fid_cf.item(),
            "lpips": lpips_score.item(),
        }
        print(all_scores)
        with open(os.path.join(output_dir, f"prelim2_{attr}_metrics_val_{args.val}.json"), "w") as f:
            json.dump(all_scores, f)
        print()


if __name__ == "__main__":
    main()