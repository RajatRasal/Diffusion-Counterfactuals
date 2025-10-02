import argparse
import multiprocessing
import json
import os
import random
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy
from torchmetrics.functional import auroc
from tqdm import tqdm

from counterfactuals.scm.image_mechanism.diffae import DiffusionImageMechanism
from models.diffae.diffae.interface import DiffusionAutoEncodersInterface
from mammo_artifacts.artifact_detector_model import Multilabel_ArtifactDetector
from mammo_artifacts.dataset import EMBEDMammoDataModule


@torch.no_grad()
def metrics(scm, dl, classifier, last_batch=-1):
    cfs = []
    study_id = []
    image_id = []
    acc_metric_circle = Accuracy(task="binary").to(classifier.device)
    acc_metric_triangle = Accuracy(task="binary").to(classifier.device)

    # Compute all metrics
    for batch_no, data in tqdm(enumerate(dl), total=len(dl)):
        print("batch no:", batch_no)
        study_id.extend(data["study_id"])
        image_id.extend(data["image_id"])
        # setup data
        factual_batch = {"image": data["image"].cuda(), "metadata": data["label"].cuda()}
        factual_batch["image"] = factual_batch["image"]
        # counterfactual
        print("cf")
        interv = factual_batch["metadata"].clone()
        interv[:, [1, 2]] = 0
        noise = scm.abduct(factual_batch)
        cf = scm.predict(noise, {"metadata": interv})["image"]
        cfs.append(cf)
        # effectiveness
        pred = classifier(cf / 2 + 0.5)
        acc_metric_circle.update(pred[:, 0], interv[:, 1])
        acc_metric_triangle.update(pred[:, 1], interv[:, 2])

        if batch_no == last_batch:
            break
    
    return torch.cat(cfs, dim=0), study_id, image_id, acc_metric_circle.compute(), acc_metric_triangle.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--last-batch", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--edit", type=str, choices=["triangle", "circle"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    # Load image model
    model_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/models/diffae/output/EMBED_192_img_512_sem_p_0.1_final"
    # model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_950_ckpt.pth'}, 'test')
    model = DiffusionAutoEncodersInterface({'output': model_dir, 'model_ckpt': 'epoch_2400_ckpt.pth'}, 'test')
    _ = model.model.eval()
    print(args, model_dir)
    image_mechanism = DiffusionImageMechanism(model, guidance_scale=args.guidance_scale)
    output_dir = os.path.join(model_dir, f"artefact_removal_{args.guidance_scale}_{args.edit}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if Path("/data2/mb121/EMBED/images/png/1024x768").exists():
        embed_data_dir = "/data2/mb121/EMBED/images/png/1024x768"
    elif Path("/data/EMBED/images/png/1024x768").exists():
        embed_data_dir = "/data/EMBED/images/png/1024x768"
    else:
        embed_data_dir = "/vol/biomedic3/data/EMBED/images/png/1024x768"

    df1 = pd.read_csv("/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv")
    df2 = pd.read_csv("/vol/biomedic3/rrr2417/cxr-generation/src/mammo_artifacts/predicted_all_embed.csv")
    df2["image_path"] = df2["image_path"].apply(lambda row: row.replace("/data/EMBED/images/png/1024x768/", ""))
    data = df1.merge(df2, on="image_path")
    data["image_path"] = data["image_path"].apply(lambda row: os.path.join(embed_data_dir, row))
    MARKER_NAMES = ["circle marker", "triangle marker"]
    data["multilabel_markers"] = data.apply(
        lambda row: np.array([row[name] for name in MARKER_NAMES]), axis=1
    )
    # print(np.stack(data.multilabel_markers.values))
    # print((np.stack(data.multilabel_markers.values)[:, 0] == 1).sum())
    # exit()

    dm = EMBEDMammoDataModule(
        target="artifact_cancer_density",
        csv_file=data,
        image_size=(192, 192),
        weighted_sampling=False,
        batch_alpha=0,
        batch_size=32,
        num_workers=1,
        return_dict=True,
        cache_size=10,
        normalise=True,
        train_augment=False,
    )
    
    if args.split == "train":
        dataset = dm.train_set
    elif args.split == "val":
        dataset = dm.val_set
    else:
        dataset = dm.test_set

    # Test dataset
    edit_map = {"circle": 1, "triangle": 2}
    idxs = np.where(dataset.labels[:, edit_map[args.edit]] == 1)[0]
    print(len(idxs))
    dl = DataLoader(Subset(dataset, idxs), batch_size=args.batch_size)

    # Classifier
    classifier_dir = "/vol/biomedic3/rrr2417/cxr-generation/src/mammo_artifacts/output/artifact-detector/version_42/checkpoints/epoch=37-step=17632.ckpt"
    classifier = Multilabel_ArtifactDetector.load_from_checkpoint(classifier_dir).cuda().eval()

    # Calculate metrics
    cfs, study_ids, image_ids, acc_circle, acc_triangle = metrics(image_mechanism, dl, classifier, args.last_batch)
    all_scores = {"acc_circle": acc_circle.item(), "acc_triangle": acc_triangle.item()}
    with open(os.path.join(output_dir, f"0_artefact_removal_{args.split}.json"), "w") as f:
        json.dump(all_scores, f)
    print(all_scores)
    
    # Save images
    new_image_ids = []
    for study_id in tqdm(study_ids):
        os.makedirs(os.path.join(output_dir, study_id), exist_ok=True)
    for image_id, cf in tqdm(zip(image_ids, cfs), total=cfs.shape[0]):
        image_id = image_id.replace(embed_data_dir, output_dir)
        new_image_ids.append(image_id)
        image = torchvision.transforms.functional.to_pil_image((cf / 2 + 0.5).clamp(0, 1))
        image.save(image_id, "PNG")
    # torch.save(cfs, os.path.join(output_dir, f"artefact_removal_{args.edit}.tensor"))
    print()
    
    df = pd.DataFrame(
        np.column_stack([new_image_ids, study_ids]),
        columns=['image_path', 'study_ids'],
    )
    df["circle marker"] = 0
    df["triangle_marker"] = 0
    df.to_csv(os.path.join(output_dir, f"edits_{args.edit}_{args.split}.csv"))


if __name__ == "__main__":
    main()