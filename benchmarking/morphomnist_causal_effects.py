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
def metrics(scm, test_dataloader, feat, classifier, last_batch=-1):
    comps = []
    acc = {
        # "de": Accuracy(task="multiclass", num_classes=10).to(classifier.device),
        # "ie": Accuracy(task="multiclass", num_classes=10).to(classifier.device),
        "te": Accuracy(task="multiclass", num_classes=10).to(classifier.device),
    }
    # cfs = {"de": [], "ie": [], "te": []}
    # revs = {"de": [], "ie": [], "te": []}
    cfs = {"te": []}
    revs = {"te": []}

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
        effects = scm.counterfactual_effects(factual_batch, {feat: interv_batch[feat]})
        # for ce_name in ["de", "ie", "te"]:
        for ce_name in ["te"]:
            cf = {"image": effects[ce_name], **effects[f"{ce_name}_parents"]} 
            cfs[ce_name].append(cf)
            # digit effectiveness
            class_preds = classifier.predict_step({"image": cf["image"].clamp(-1, 1)}).argmax(dim=1)
            class_targets = cf["class"].argmax(dim=1)        
            acc[ce_name].update(class_preds, class_targets)
            # reversibility
            cf_rev = scm.counterfactual(cf, factual_batch) 
            rev = l1_distance(cf_rev["image"], factual_batch["image"])
            revs[ce_name].append(rev)
        # end if we only want to test on a subset of data
        if batch_no == last_batch:
            break

    # Aggregate comp and rev
    comp = torch.cat(comps, dim=0).mean()
    revs = {k: torch.cat(v, dim=0).mean() for k, v in revs.items()}

    def mape(gt, pred):
        ape = torch.abs((gt - pred) / gt)
        mape = torch.median(ape)  #.mean()
        return mape
    
    # Effectiveness for other attributes
    eff = {}
    for ce_name, cf in cfs.items():
        cfs_agg = defaultdict(list)
        for _cf in cf:
            for k, v in _cf.items():
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
        eff_metrics["class"] = acc[ce_name].compute()
        eff[ce_name] = eff_metrics
    
    return comp, revs, eff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["diffusion", "vae", "hvae"], default="diffusion")
    parser.add_argument("--version", type=str, required=True)
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
        # folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_ddpm_mnist/"
        # model_folder = f"lightning_logs/{args.version}"
        # hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
        # with open(hparams_yaml, "r") as f:
        #     hparams = yaml.safe_load(f.read())
        # model = GuidedDiffusion(**hparams).cuda()
        # ckpt_path = os.path.join(folder, model_folder, "checkpoints/last.ckpt")
        # weights = torch.load(ckpt_path)
        # model.load_state_dict(weights["state_dict_ema"])

        # # p_uncond_idx = args.version.index("p")
        # # cfg = args.version[p_uncond_idx + 2:p_uncond_idx + 5] > 0
        # # print("CFG:", args.cfg)
        # image_mechanism = DiffusionImageMechanism(
        #     model,
        #     args.guidance_scale if args.cfg else 0,
        #     args.timesteps,
        #     "ddim",
        #     "cfg" if args.cfg else "ddim",
        #     include_d=False,
        # )
        # output_dir = os.path.join(folder, model_folder, f"cfg_{args.cfg}_g_{args.guidance_scale}_t_{args.timesteps}")

        if args.semantic:
            if args.cfg:
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_p_0.5_pa_cond"
            else:
                model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_pa_cond"
        else:
            if args.cfg:
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_ddim_cfg_4'
            else:
                model_dir = '/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_ddim_cond_4'
                # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_ch_64_cosine"
        model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_300_ckpt.pth'}, 'test')
        _ = model.model.eval()
        hparams = {"seed": model.cfg["general"]["seed"]}
        print("---------------", args)
        guidance_scale = args.guidance_scale if args.cfg else 1
        image_mechanism = DiffusionImageMechanism(model, guidance_scale=guidance_scale)
        output_dir = os.path.join(model_dir, f"cfg_{args.cfg}_g_{guidance_scale}")
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
        ckpt_path = os.path.join(folder, model_folder, "checkpoints/model_epoch=99.ckpt")  # "checkpoints/last.ckpt")
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
    for feat in ["slant", "class"]:  # features_order:
        print("Intervention:", feat)
        comp, rev, eff = metrics(scm, dl, feat, classifier, args.last_batch)
        all_scores = {
            "comp": comp.item(),
            "rev": {k: v.item() for k, v in rev.items()},
            "eff": {ce_name: {k: v.item() for k, v in ce.items()} for ce_name, ce in eff.items()}
        }
        print(all_scores)
        with open(os.path.join(output_dir, f"{feat}_metrics.json"), "w") as f:
            json.dump(all_scores, f)


if __name__ == "__main__":
    main()