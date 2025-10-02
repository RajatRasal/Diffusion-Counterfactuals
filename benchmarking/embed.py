import argparse
import multiprocessing
import json
import os
import random
import yaml
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy
from torchmetrics.functional import auroc
from tqdm import tqdm

from counterfactuals.scm.image_mechanism.diffae import DiffusionImageMechanism
from models.diffae.diffae.interface import DiffusionAutoEncodersInterface
from mammo_artifacts.artifact_detector_model import Multilabel_ArtifactDetector


@torch.no_grad()
def metrics(scm, test_dataloader, feat, classifier, last_batch=-1):
    cfs = []
    intervs = []
    acc_metric_circle = Accuracy(task="binary").to(classifier.device)
    acc_metric_triangle = Accuracy(task="binary").to(classifier.device)

    # Compute all metrics
    for batch_no, (img, cond) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        print("batch no:", batch_no)
        # setup data
        factual_batch = {"image": img.cuda(), "metadata": cond.cuda()}
        factual_batch["image"] = factual_batch["image"]
        # counterfactual
        print("cf")
        interv = factual_batch["metadata"].clone()
        interv[:, feat] = (~interv[:, feat].bool()).float()
        intervs.append(interv[:, feat])
        noise = scm.abduct(factual_batch)
        cf = scm.predict(noise, {"metadata": interv})["image"]
        cfs.append(cf)
        # effectiveness
        pred = classifier(cf / 2 + 0.5)
        acc_metric_circle.update(pred[:, 0], interv[:, 1])
        acc_metric_triangle.update(pred[:, 1], interv[:, 2])

        if batch_no == last_batch:
            break

    return torch.cat(cfs, dim=0), acc_metric_circle.compute(), acc_metric_triangle.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--edit", type=str, default="triangle")
    args = parser.parse_args()

    # Load image model
    model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/EMBED_192_img_512_sem_p_0.1_final"
    # model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/EMBED_192_img_512_no_sem_p_0.5_final"
    # model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_950_ckpt.pth'}, 'test')
    model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_2400_ckpt.pth'}, 'test')
    _ = model.model.eval()
    print(args, model_dir)
    image_mechanism = DiffusionImageMechanism(model, guidance_scale=args.guidance_scale)
    output_dir = os.path.join(model_dir, f"metrics_g_{args.guidance_scale}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test dataset
    edit_map = {"circle": 1, "triangle": 2}
    edit_idx = edit_map[args.edit]
    idxs = np.where(model.dataset.labels[:, edit_idx] == 1)[0]
    dl = DataLoader(Subset(model.dataset, idxs), batch_size=args.batch_size)

    # Classifier
    classifier_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/mammo_artifacts/output/artifact-detector/version_42/checkpoints/epoch=37-step=17632.ckpt"
    classifier = Multilabel_ArtifactDetector.load_from_checkpoint(classifier_dir).cuda().eval()

    # Calculate metrics
    print("Intervention:", args.edit)
    cfs, acc_circle, acc_triangle = metrics(image_mechanism, dl, edit_idx, classifier, args.last_batch)
    all_scores = {"acc_circle": acc_circle.item(), "acc_triangle": acc_triangle.item()}
    print(all_scores)
    torch.save(cfs, os.path.join(output_dir, f"3_{args.edit}.tensor"))
    with open(os.path.join(output_dir, f"3_{args.edit}.json"), "w") as f:
        json.dump(all_scores, f)
    print()


if __name__ == "__main__":
    main()