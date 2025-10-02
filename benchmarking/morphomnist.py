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
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from classification.mnist import MNISTClassifierLightningModule
from counterfactuals.benchmarking.composition import composition
from counterfactuals.benchmarking.l1_distance import l1_distance
from counterfactuals.benchmarking.reversibility import reversibility
from counterfactuals.scm.image_mechanism.mechanisms import DiffusionImageMechanism
from counterfactuals.scm.morphomnist.numerical_mechanisms import NumericalMorphoMNISTMechanisms
from counterfactuals.scm.full_scm import SCM
from data.morphomnist.datamodules import MorphoMNISTDataModule
from data.morphomnist.datasets import MorphoMNIST
from ddpm.training.mnist_conditional import GuidedDiffusionLightningModule
from morphomnist.morphomnist.measure import measure_batch


@torch.no_grad
def _composition(scm, test_dataloader, cycles, fast=False):
    all_composition_scores = []
    all_composition_images = []
    for idx, batch in tqdm(enumerate(test_dataloader)):
        batch = {k: v.cuda() for k, v in batch.items()}
        scores, images = composition(scm, batch, cycles)
        all_composition_scores.append(
            torch.cat([x.unsqueeze(0) for x in scores], dim=0)
        )
        all_composition_images.append(
            torch.cat([x.unsqueeze(0) for x in images], dim=0)
        )
        if idx == 1 and fast:
            break
    return torch.cat(all_composition_scores, dim=1), torch.cat(all_composition_images, dim=1)


@torch.no_grad
def _reversibility(scm, test_dataloader, cycles, interv_features, seed=0, fast=False):
    random.seed(seed)
    all_reversibilty_scores = []
    all_reversibilty_chain = []
    for idx, batch in tqdm(enumerate(test_dataloader)):
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda() for k, v in batch.items()}
        interv_batch = {k: v[idxs] for k, v in factual_batch.items() if k in interv_features}
        scores, chain = reversibility(scm, factual_batch, interv_batch, cycles)
        all_reversibilty_scores.append(torch.cat([x.unsqueeze(0) for x in scores], dim=0))
        all_reversibilty_chain.append(torch.cat([x.unsqueeze(0) for x in chain], dim=0))
        if idx == 1 and fast:
            break
    return torch.cat(all_reversibilty_scores, dim=1), torch.cat(all_reversibilty_chain, dim=1)


@torch.no_grad
def _effectiveness(scm, test_dataloader, interv_features, classifier, seed=0, fast=False):
    random.seed(seed)
    cfs = []
    factual_images = []
    fid_metric = FrechetInceptionDistance(feature=64, normalize=True, compute_on_cpu=True, input_img_size=(3, 28, 28)).to(classifier.device)
    acc_metric = Accuracy(task="multiclass", num_classes=10).to(classifier.device)
    for idx, batch in tqdm(enumerate(test_dataloader)):
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {k: v.cuda() for k, v in batch.items()}
        factual_images.append(factual_batch["image"])
        interv_batch = {k: v[idxs] for k, v in factual_batch.items() if k in interv_features}
        cf_batch = scm.counterfactual(factual_batch, interv_batch)
        cfs.append(cf_batch)
        class_preds = classifier.predict_step(cf_batch).argmax(dim=1)
        fid_metric.update(scm.image_mechanism.model._postprocess_images(factual_batch["image"]).repeat(1, 3, 1, 1) * 255, real=True)
        fid_metric.update(scm.image_mechanism.model._postprocess_images(cf_batch["image"]).repeat(1, 3, 1, 1) * 255, real=False)
        acc_metric.update(class_preds, cf_batch["class"].argmax(dim=1))
        if idx == 1 and fast:
            break
    
    cfs_agg = defaultdict(list)
    for cf in cfs:
        for k, v in cf.items():
            cfs_agg[k].append(v)
    cfs_agg = {k: torch.cat(v, dim=0) for k, v in cfs_agg.items()}

    cf_images_unnormalised = scm.image_mechanism.model._postprocess_images(cfs_agg["image"]) * 255
    with multiprocessing.Pool() as pool:
        test_metrics = measure_batch(cf_images_unnormalised.cpu().numpy().squeeze(1), pool=pool)
    test_metrics = {
        metric: torch.nn.functional.l1_loss(
            MorphoMNIST.unnormalise_metadata_helper(metric, cfs_agg[metric].cpu()),
            torch.tensor(test_metrics[metric].to_numpy())
        ).item()
        for metric in ["thickness", "slant", "intensity"]
    }
    test_metrics["class"] = acc_metric.compute().item()
    test_metrics["fid"] = fid_metric.compute().item()

    return test_metrics, cfs_agg["image"], factual_images


@torch.no_grad
def _realism(counterfactual_images, factual_images):
    FID()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=136)
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--timestep-spacing", type=str, default="leading")
    args = parser.parse_args()

    # Load image model
    folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_train/"
    # folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_cf_finetune/"
    model_folder = f"lightning_logs/version_{args.version}"
    hparams_yaml = os.path.join(folder, model_folder, "hparams.yaml")
    with open(hparams_yaml, "r") as f:
        hparams = yaml.safe_load(f.read())
    # hparams["timestep_spacing"] = args.timestep_spacing
    model = GuidedDiffusionLightningModule(**hparams).cuda()
    ckpt_path = os.path.join(
        folder,
        model_folder,
        "checkpoints",
        "last.ckpt" if args.last else f"model_epoch={args.epoch}.ckpt",
    )
    weights = torch.load(ckpt_path)
    model.load_state_dict(weights["state_dict_ema"])

    # Create output directory
    epoch_name = "last" if args.last else str(args.epoch)
    output_dir = os.path.join(folder, model_folder, f"g_{args.guidance_scale}_t_{args.timesteps}_{epoch_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Load image mechanism
    do_cfg = args.guidance_scale != 1.
    numerical_mechanisms = NumericalMorphoMNISTMechanisms(normalised=True, device=torch.device('cuda'))
    image_mechanism = DiffusionImageMechanism(model, args.guidance_scale, args.timesteps)
    scm = SCM(image_mechanism, numerical_mechanisms)

    # Test dataset
    data_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/data/morphomnist/files/full_dataset2/0"
    dm_test = MorphoMNISTDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        pin_memory=True,
        split_ratio=(0.9, 0.1),
        num_workers=0,
        prefetch_factor=None,
        normalise_metadata=True,
    )
    dm_test.setup("test")
    test_dataloader = dm_test.test_dataloader()

    # Classifier
    classifier = MNISTClassifierLightningModule.load_from_checkpoint(
        "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_classifier/lightning_logs/version_55/checkpoints/last.ckpt"
    )

    return output_dir, model, classifier, scm, test_dataloader, args.fast


def main():
    output_dir, model, classifier, scm, test_dataloader = _main()

    cycles = 10

    # Composition
    print("Composition")
    all_composition_scores, all_composition_images = _composition(scm, test_dataloader, cycles, fast=args.fast)
    comp_1 = all_composition_scores[0].mean()
    comp_10 = all_composition_scores[9].mean()
    print(f"composition 1: {comp_1}")
    print(f"composition 10: {comp_10}")

    fig, axes = plt.subplots(cycles + 1, cycles + 1, figsize=(3, 3))
    for i in range(cycles + 1):
        for j in range(cycles + 1):
            axes[i, j].imshow(all_composition_images[j, i, 0].cpu().numpy(), cmap="grey", vmin=-1, vmax=1)
            axes[i, j].set_axis_off()
    fig.tight_layout(pad=-0.1)
    fig.savefig(os.path.join(output_dir, "composition.pdf"), bbox_inches="tight")

    # Reversiblity
    print("Reversiblity")
    all_reversibility_scores, all_reversibility_images = _reversibility(
        scm, test_dataloader, cycles, ["class", "intensity", "thickness", "slant"], fast=args.fast,
    )
    rev_1 = all_reversibility_scores[0].mean()
    rev_10 = all_reversibility_scores[9].mean()
    print(f"reversibility 1: {rev_1}")
    print(f"reversibility 10: {rev_10}")

    fig, axes = plt.subplots(cycles + 1, cycles + 1, figsize=(3, 3))
    for i in range(cycles + 1):
        for j in range(cycles + 1):
            axes[i, j].imshow(all_reversibility_images[j, i, 0].cpu().numpy(), cmap="grey", vmin=-1, vmax=1)
            axes[i, j].set_axis_off()
    fig.tight_layout(pad=-0.1)
    fig.savefig(os.path.join(output_dir, "reversibility.pdf"), bbox_inches='tight')

    # # Effectiveness
    # print("Effectiveness")
    # avg_effectiveness_scores, cf_images, factual_images = _effectiveness(scm, test_dataloader, ["class", "thickness"], classifier, fast=args.fast)

    # print(len(cf_images))
    # print(len(factual_images))
    # print(avg_effectiveness_scores)

    # fig, axes = plt.subplots(11, 11, figsize=(3, 3))
    # for ax in axes.flatten():
    #     ax.set_axis_off()
    # for orig, ax in zip(factual_images[args.batch_size:], axes[:, [0, 4, 8]].flatten()):
    #     ax.imshow(orig.cpu().numpy()[0], cmap="grey", vmin=-1, vmax=1)
    # for cf, ax in zip(cf_images[args.batch_size:], axes[:, [2, 6, 10]].flatten()):
    #     ax.imshow(cf.cpu().numpy()[0], cmap="grey", vmin=-1, vmax=1)
    # for comp, ax in zip(all_composition_images[1][args.batch_size:], axes[:, [1, 5, 9]].flatten()):
    #     ax.imshow(comp.cpu().numpy()[0], cmap="grey", vmin=-1, vmax=1)
    # fig.tight_layout(pad=-0.1)
    # fig.savefig(os.path.join(output_dir, "effectiveness.pdf"), bbox_inches="tight")
    # plt.show()

    # Store metrics
    all_scores = {
        # **{f"eff_{k}": v for k, v in avg_effectiveness_scores.items()},
        "comp_1": comp_1.item(),
        "comp_10": comp_10.item(),
        "rev_1": rev_1.item(),
        "rev_10": rev_10.item(),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_scores, f)


def single_intervention_effectiveness():
    output_dir, model, classifier, scm, test_dataloader, fast = _main()

    output_dir = os.path.join(output_dir, "effectiveness")
    os.makedirs(output_dir, exist_ok=True)

    print("Effectiveness")
    classifier = MNISTClassifierLightningModule.load_from_checkpoint(
        "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_classifier/lightning_logs/version_55/checkpoints/last.ckpt"
    )

    all_scores = {}
    for feat in ["intensity", "thickness", "class", "slant"]:
        print(feat)
        avg_effectiveness_scores, cf_images, factual_images = _effectiveness(scm, test_dataloader, [feat], classifier, fast=fast)
        print(avg_effectiveness_scores)
        all_scores[feat] = avg_effectiveness_scores

    with open(os.path.join(output_dir, "effectiveness.json"), "w") as f:
        json.dump(all_scores, f)


if __name__ == "__main__":
    single_intervention_effectiveness()
    # main()