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
from morphomnist.morphomnist.measure import measure_batch


@torch.no_grad()
def metrics(scm, test_dataloader, feat, classifier, seed, last_batch=-1):
    comps = []
    revs = []
    acc = Accuracy(task="multiclass", num_classes=10).to(classifier.device) 
    cfs = []

    random.seed(seed)

    # Compute all metrics
    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        print("Batch no:", batch_no)
        # setup data
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda().clone() for k, v in batch.items()}
        interv_batch = {k: v.clone()[idxs] if k == feat else v.clone() for k, v in factual_batch.items()}
        # composition
        print("Composition")
        noise = scm.abduct(factual_batch)
        cf_comp = scm.predict(noise, factual_batch)
        comp = l1_distance(cf_comp["image"].clamp(-1, 1), factual_batch["image"])
        print(cf_comp["image"].shape)
        comps.append(comp)
        # counterfactual
        print("Counterfactual")
        cf = scm.predict(noise, {feat: interv_batch[feat]})
        cfs.append(cf)
        class_preds = classifier.predict_step({"image": cf["image"].clamp(-1, 1)})
        class_targets = cf["class"].argmax(dim=1)
        acc.update(class_preds, class_targets)
        # reversibility
        print("Reversibility")
        cf_rev = scm.counterfactual(cf, factual_batch)
        rev = l1_distance(cf_rev["image"].clamp(-1, 1), factual_batch["image"])
        revs.append(rev)
        # end if we only want to test on a subset of data
        if batch_no == last_batch:
            break

    # Aggregate comp and rev
    comp = torch.cat(comps, dim=0).mean()
    revs = torch.cat(revs, dim=0).mean()

    def mape(gt, pred):
        print(gt, pred)
        ape = torch.abs((gt - pred) / gt)
        mape = ape.mean()
        return mape
    
    # Effectiveness for other attributes
    cfs_agg = defaultdict(list)
    for cf in cfs:
        for k, v in cf.items():
            cfs_agg[k].append(v)
    cfs_agg = {k: torch.cat(v, dim=0) for k, v in cfs_agg.items()}
    cf_images = (cfs_agg["image"].clamp(-1, 1) / 2 + 0.5) * 255
    print(cf_images.shape, cf_images.min(), cf_images.max())
    with multiprocessing.Pool() as pool:
        measures = measure_batch(cf_images.cpu().numpy().squeeze(1), pool=pool)
    eff_metrics = {
        metric: mape(
            MorphoMNIST.unnormalise_metadata_helper(metric, cfs_agg[metric].cpu()),
            torch.tensor(measures[metric].to_numpy()),
        )
        for metric in ["thickness", "slant", "intensity"]
    }
    eff_metrics["class"] = acc.compute()
    print(eff_metrics)
    
    return comp, revs, eff_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["diffusion", "vae", "hvae"], default="diffusion")
    # parser.add_argument("--version", type=str, required=True)
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
    if args.model == "diffusion":
        if args.semantic:
            if args.cfg:
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_p_0.1"
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_ch_64_p_0.1_dim_8"
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_ch_64_p_0.1_dim_4"
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_cfg_4"
            else:
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_ch_64_p_0_dim_8"
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_cond_4"
        else:
            if args.cfg:
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_ddim_cfg_4'
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_p_0.1"
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_p_0.5"
            else:
                # model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_ddim_cond_4'
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_cosine"
        print(model_dir)
        model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_500_ckpt.pth'}, 'test')
        _ = model.model.eval()
        hparams = {"seed": model.cfg["general"]["seed"]}
        print("---------------", args)
        guidance_scale = args.guidance_scale if args.cfg else 1
        image_mechanism = DiffusionImageMechanism(model, guidance_scale=guidance_scale)
        output_dir = os.path.join(model_dir, f"prelim_sem_trial_2_{args.semantic}_cfg_{args.cfg}_g_{guidance_scale}")
    elif args.model == "vae":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_vae_mnist"
        model_folder = f"lightning_logs/version_27"
        hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        with open(hparams_yaml, "r") as f:
            hparams = yaml.safe_load(f.read())
        model = ConditionalVAE(**hparams).cuda()
        hparams["seed"] = args.seed
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights["state_dict_ema"])
        output_dir = os.path.join(folder, model_folder, "prelim_metrics")
        image_mechanism = VAEImageMechanism(model=model, deterministic=False)
    elif args.model == "hvae":
        folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_hvae_mnist"
        model_folder = f"lightning_logs/version_14"
        hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        with open(hparams_yaml, "r") as f:
            hparams = yaml.safe_load(f.read())
        model = ConditionalHVAE(**hparams).cuda()
        hparams["seed"] = args.seed
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights["state_dict_ema"])
        output_dir = os.path.join(folder, model_folder, "prelim_metrics")
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
    for feat in features_order:  # ["intensity"]:  # "slant", "class"]: # features_order:
        print("Intervention:", feat)
        comp, rev, eff = metrics(scm, dl, feat, classifier, args.seed, args.last_batch)
        all_scores = {
            "comp": comp.item(),
            "rev": rev.item(),  # {k: v.item() for k, v in rev.items()},
            "eff": {k: v.item() for k, v in eff.items()},  # {ce_name: {k: v.item() for k, v in ce.items()} for ce_name, ce in eff.items()}
        }
        print(all_scores)
        with open(os.path.join(output_dir, f"{feat}_metrics.json"), "w") as f:
            json.dump(all_scores, f)


if __name__ == "__main__":
    main()