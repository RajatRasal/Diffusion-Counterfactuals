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

        print("Batch no:", batch_no)
        idxs = list(range(batch["image"].shape[0]))
        random.shuffle(idxs)
        factual_batch = {"image": batch["image"].cuda().clone(), "metadata": batch["metadata"].cuda().clone()}
        interv_batch = {"metadata": batch["metadata"][idxs].cuda().clone()}

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
        plt.savefig(f"./grid_digit_diff.png", bbox_inches="tight")
        comp = l1_distance((cf_comp["image"] + 1) / 2, (factual_batch["image"] + 1) / 2)
        comps.append(comp)

        # counterfactual
        print("Counterfactual")
        cf = scm.predict(noise, interv_batch)
        grid = torchvision.utils.make_grid(cf["image"][:32], 4)
        img = F_vision.to_pil_image(grid)
        plt.imshow(np.asarray(img))
        plt.savefig(f"./grid_cf_digit_diff.png", bbox_inches="tight")
        cfs.append(cf)
        class_preds = classifier.predict_step({"image": cf["image"]}).argmax(dim=1)
        acc.update(class_preds, interv_batch["metadata"].argmax(dim=1))

        # reversibility
        # print("Reversibility")
        print(cf.keys())
        cf_rev = scm.predict(scm.abduct({**cf, "metadata": interv_batch["metadata"]}), factual_batch)
        rev = l1_distance((cf_rev["image"] + 1) / 2, (factual_batch["image"] + 1) / 2)
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
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    # parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    classifier = MNISTClassifier.load_from_checkpoint(
        "/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_classifier_mnist/lightning_logs/version_0/checkpoints/last.ckpt",
    )

    # Load image model
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_d"  # 500
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_d2"  # 500
    model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_sem_d3"  # 500
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/mnist_cond_d"  # 200
    print(model_dir)
    model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_500_ckpt.pth'}, 'test')
    _ = model.model.eval()
    hparams = {"seed": model.cfg["general"]["seed"]}
    print("---------------", args)
    guidance_scale = args.guidance_scale
    image_mechanism = DiffusionImageMechanism(model, guidance_scale=guidance_scale)
    output_dir = os.path.join(model_dir, f"prelim_sem_trial_2_{args.semantic}_cfg_{args.cfg}_g_{guidance_scale}")

    # Test dataset
    data_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/data/morphomnist/files/s_i_t_d"
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
        features=["class"],
        data_dir=data_dir,
    )
    dm.setup("test")
    dl = dm.test_dataloader()

    # Calculate metrics
    comp, rev, acc = metrics(image_mechanism, dl, classifier, args.seed, args.last_batch)
    all_scores = {
        "comp": comp.item(),
        "rev": rev.item(),
        "acc": acc.item(),
    }
    print(all_scores)


if __name__ == "__main__":
    main()