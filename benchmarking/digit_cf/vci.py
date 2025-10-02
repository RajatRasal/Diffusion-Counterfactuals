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
import numpy as np

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
from counterfactuals.scm.image_mechanism.vci import VCIImageMechanism
from models.variational_causal_inference.vci.train.prepare import prepare

import torch.nn.functional as F


@torch.no_grad()
def metrics(scm, test_dataloader, classifier, seed, last_batch=-1):
    comps = []
    revs = []
    acc = Accuracy(task="multiclass", num_classes=10).to(classifier.device)
    cfs = []
    cf_errors = []
    interv_labels = []

    cf_errors1 = []
    cf_errors2 = []

    random.seed(seed)

    # Compute all metrics
    for batch_no, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        image, metadata, _, _, _, label = batch

        print("Batch no:", batch_no)
        idxs = list(range(image.shape[0]))
        random.shuffle(idxs)
        factual_batch = {"image": image.cuda().clone(), "metadata": metadata.cuda().clone()}
        interv_batch = {"metadata": metadata[idxs].cuda().clone()}

        # interv_batch = {"metadata": metadata.cuda().clone()}
        # interv_label = {k: torch.tensor(v).cuda().clone() for k, v in label.items()}

        # composition
        # print("Composition")
        noise = scm.abduct(factual_batch)
        cf_comp = scm.predict(noise, factual_batch)
        import torchvision
        import torchvision.transforms.functional as F_vision
        import matplotlib.pyplot as plt
        grid = torchvision.utils.make_grid(cf_comp["image"][:32], 4)
        img = F_vision.to_pil_image(grid)
        plt.imshow(np.asarray(img))
        plt.savefig(f"./grid_digit.png", bbox_inches="tight")
        comp = l1_distance(cf_comp["image"][:, :, 2:-2, 2:-2], factual_batch["image"][:, :, 2:-2, 2:-2])
        comps.append(comp)

        # counterfactual
        print("Counterfactual")
        cf = scm.predict(noise, interv_batch)
        grid = torchvision.utils.make_grid(cf["image"][:32, :, 2:-2, 2:-2], 4)
        img = F_vision.to_pil_image(grid)
        plt.imshow(np.asarray(img))
        plt.savefig(f"./grid_cf_digit.png", bbox_inches="tight")
        cfs.append(cf)
        print(cf["image"].shape)
        class_preds = classifier.predict_step({"image": (cf["image"] * 2 - 1)[:, :, 2:-2, 2:-2]}).argmax(dim=1)
        print(class_preds.shape)
        acc.update(class_preds, torch.tensor([int(l) for l in label["label"]]).cuda()[idxs])

        # reversibility
        # print("Reversibility")
        cf_rev = scm.predict(scm.abduct(cf), factual_batch)
        rev = l1_distance(cf_rev["image"][:, :, 2:-2, 2:-2], factual_batch["image"][:, :, 2:-2, 2:-2])
        revs.append(rev)
        # end if we only want to test on a subset of data
        if batch_no == last_batch:
            break

    # Aggregate comp and rev
    comp = torch.cat(comps, dim=0).mean()
    revs = torch.cat(revs, dim=0).mean()
    acc = acc.compute()

    return comp, revs, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    classifier = MNISTClassifier.load_from_checkpoint(
        "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_classifier_mnist/lightning_logs/version_0/checkpoints/last.ckpt",
    )

    ckpt = "/vol/biomedic3/rrr2417/cxr-generation/src/models/variational_causal_inference/artifacts/saves/morphoMNIST-digit-test_2025.03.27_14:51:40/model_seed=None_epoch=299.pt"
    state_dict, args2 = torch.load(ckpt, map_location="cuda") 
    args2["batch_size"] = args.batch_size
    model, datasets = prepare(args2, state_dict, device="cuda")
    dl_test = datasets["test_loader"]

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # SCM
    image_mechanism = VCIImageMechanism(model, True)

    # Calculate metrics
    comp, rev, acc = metrics(image_mechanism, dl_test, classifier, args.seed, args.last_batch)
    all_scores = {
        "comp": comp.item(),
        "rev": rev.item(),
        "acc": acc.item(),
    }
    print(all_scores)


if __name__ == "__main__":
    main()