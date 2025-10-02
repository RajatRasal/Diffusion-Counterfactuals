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
from torchmetrics.functional.image.lpips import _NoTrainLpips
from tqdm import tqdm

from counterfactuals.scm.full_scm import SCM
from counterfactuals.scm.image_mechanism.diffae_cta import COImageMechanism
from counterfactuals.scm.numerical_mechanism.celeba import CelebANumericalMechanisms
from data.mixer import datamodule_mixer
from models.classifier.cnn import ResNetBinaryClassifier
from models.diffae.diffae.interface import DiffusionAutoEncodersInterface


def metrics(scm, test_dataloader, feat, classifier_glasses, classifier_smiling, last_batch=-1):
    comps = []
    cfs = []
    interv_glasses = []
    interv_smiling = []

    # Compute all metrics
    for image_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        if image_no > 22:
            break
        if image_no not in [9, 17, 15, 21]:
            continue
        print("image no:", image_no)
        # setup data
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}
        # fitting
        noise = scm.abduct(factual_batch)
        # composition
        print("composition")
        cf_comp = scm.predict(noise, factual_batch)
        comps.append(cf_comp["image"])
        # counterfactual
        print("cf")
        cf = scm.predict(noise, {feat: (~factual_batch[feat].bool()).float()})
        cfs.append(cf["image"])
        interv_glasses.append(cf["Eyeglasses"])
        interv_smiling.append(cf["Smiling"])
        if image_no == last_batch:
            break

    # effectiveness
    print(cfs[0].shape)
    cfs = torch.cat(cfs, dim=0)
    cf_preds_glasses = classifier_glasses({"image": cfs}).detach().flatten()
    cf_preds_smiling = classifier_smiling({"image": cfs}).detach().flatten()
    interv_glasses = torch.cat(interv_glasses, dim=0).flatten()
    interv_smiling = torch.cat(interv_smiling, dim=0).flatten()
    acc_glasses = ((cf_preds_glasses > 0.5).float() == interv_glasses).float().mean()
    acc_smiling = ((cf_preds_smiling > 0.5).float() == interv_smiling).float().mean()

    # IDP
    comps = torch.cat(comps, dim=0)
    lpips_score = _NoTrainLpips(net="alex").cuda()(comps, cfs).mean().detach()

    return cfs, comps, lpips_score, acc_glasses, acc_smiling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--cta-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--last-batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    # Load image model
    if args.semantic:
        model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/sem_enc_long_train'
        model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.5'
    else:
        model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/ddim_long_train'
        # /vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/CELEBA_p_0.5_no_sem
    model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_6000_ckpt.pth'}, 'test')
    _ = model.model.eval()
    hparams = {"seed": model.cfg["general"]["seed"]}
    print(args, model_dir)
    image_mechanism = COImageMechanism(
        model,
        guidance_scale=args.guidance_scale,
        cta_steps=args.cta_steps,
        learning_rate=args.learning_rate,
    )
    output_dir = os.path.join(model_dir, f"nto_g_{args.guidance_scale}_steps_{args.cta_steps}_lr_{args.learning_rate}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test dataset
    dm, _, _, features = datamodule_mixer(
        dataset="celebahq",
        batch_size=1,
        num_workers=4,
        prefetch_factor=None,
        split_ratio=(0.9, 0.1),
        seed=hparams["seed"],
        resolution=64,
        cache=False,
        **{"horizontal_flip_prob": 0}
    )
    dm.setup("test")
    dl = dm.test_dataloader()

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
    for attr in ["Eyeglasses", "Smiling"]:
        print("Intervention:", attr)
        cfs, comps, lpips_score, acc_glasses, acc_smiling = metrics(scm, dl, attr, classifier_glasses, classifier_smiling, args.last_batch)

        all_scores = {
            "acc_g": acc_glasses.item(),
            "acc_s": acc_smiling.item(),
            "lpips_score": lpips_score.item(),
        }
        print(all_scores)
        with open(os.path.join(output_dir, f"{attr}_metrics_nto.json"), "w") as f:
            json.dump(all_scores, f)
        torch.save(cfs, os.path.join(output_dir, f"cfs_{attr}.tensor"))
        torch.save(comps, os.path.join(output_dir, f"comps_{attr}.tensor"))

        print()


if __name__ == "__main__":
    main()