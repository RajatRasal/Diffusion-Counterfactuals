"""
The codes are modified.

Link:
    - [CelebAHQ]
        - https://github.com/pytorch/vision/
          blob/677fc939b21a8893f07db4c1f90482b648b6573f/torchvision/datasets/celeba.py#L15-L189
    - [D2CCrop]
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/dataset.py#L193-L217
"""
import csv
import os
import re

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from tqdm import tqdm

from .mnist import MorphoMNIST
from mammo_artifacts.dataset import EMBEDMammoDataModule



class EmbeddingDataset(Dataset):
    """SemanticEncoder feature dataset.
    """
    def __init__(self, image_loader, encoder, cfg):
        """
        Args:
            image_loader: Pytorch DataLoader class that returns the images tensor and targets.
            encoder: SemanticEncoder module.
            cfg: A dict of config.
        """
        super().__init__()
        self.device = cfg['general']['device']
        self.style_embs, self.targets = self._extract_embedding(image_loader, encoder)

    @torch.inference_mode()
    def _extract_embedding(self, image_loader, encoder):
        encoder.to(self.device)
        encoder.eval()

        style_embs = []
        targets = []
        for _, batch in tqdm(enumerate(image_loader), total=len(image_loader), desc='Extracting features...'):
            image, target = batch
            image = image.to(self.device)
            style_emb = encoder(image)
            style_embs.append(style_emb.to('cpu'))
            targets.append(target)
        style_embs = torch.cat(style_embs)
        targets = torch.cat(targets)

        return style_embs, targets

    def __len__(self):
        return self.style_embs.shape[0]

    def __getitem__(self, index):
        style_emb = self.style_embs[index]
        target = self.targets[index]
        return style_emb, target.float()


class CelebAHQ(CelebA):
    base_folder = 'celebahq'
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ('1badu11NqxGf6qM3PTTooQDJvQbejgbTv', 'b08032b342a8e0cf84c273db2b52eef3', 'CelebAMask-HQ.zip'),
        ('0B7EVK8r0v71pY0NSMzRuSXJEVkk', 'd32c9cbf5e040fd4025c592c306e6668', 'list_eval_partition.txt'),
    ]

    def __init__(
        self,
        root,
        split='train',
        target_type='attr',
        attributes=None,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        split_ = split_map[verify_str_arg(split.lower(), 'split', ('train', 'valid', 'test', 'all'))]
        splits = self._load_csv('list_eval_partition.txt')
        id_map = self._load_id_map('CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', header=0)
        attr = self._load_csv('CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.filename = [id_map[f] for f in self.filename if f in id_map.keys()]

        self.attr = torch.zeros((len(self.filename), attr.data.shape[1]), dtype=torch.int64)
        if split_ is not None:
            for i, f in enumerate(self.filename):
                num = int(re.sub(r'[^0-9]', '', f))
                self.attr[i] = attr.data[num]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

        if attributes is not None:
            self.attr_idxs = [self.attr_names.index(name) for name in attributes]

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in ['.zip', '.7z'] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img'))

    def download(self):
        return
        # if self._check_integrity():
        #     print('Files already downloaded and verified')
        #     return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, None) #md5)

        extract_archive(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ.zip'))

    def _load_id_map(self, filename, header=None):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[2] for row in data]  # orig_file

        id_map = {}
        for idx, orig_file in zip(indices, data):
            assert isinstance(orig_file, str)
            id_map[orig_file] = f'{idx}.jpg'

        return id_map

    def __getitem__(self, index):
        x = PIL.Image.open(
            os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img', self.filename[index])
        )

        target = []
        for t in self.target_type:
            if t == 'attr':
                target.append(self.attr[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            x = self.transform(x)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        if hasattr(self, "attr_idxs"):
            target = target[self.attr_idxs]

        return x, target.float()


def get_dataset(name, split, transform=None, attributes=None):
    """Get torchvision dataset.

    Args:
        name (str): Name of one of the following datasets,
            celeba: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
            celebahq: http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
        split (str): One of [`train`, `val`, `test`].
        transform (callable): A transform function that takes in an PIL image and returns a transformed version.

    Returns:
        dataset: A dataset class.
    """
    if name == 'celeba':
        dataset = CelebA(
            root='./datasets/',
            split=split,
            target_type='attr',
            transform=transform,
            download=True,
        )
    elif name == 'celebahq':
        dataset = CelebAHQ(
            root='/vol/biomedic3/rrr2417/diffusion-autoencoders/datasets/',
            split=split,
            target_type='attr',
            transform=transform,
            attributes=attributes,
            download=True,
        )
    elif name == "mnist":
        dataset = MorphoMNIST(
            root="/vol/biomedic3/rrr2417/cxr-generation/src/data/morphomnist/files/s_i_t_d",
            train=split == "train",
            transform=transform,
            one_hot=True,
            normalise_metadata=True,
            features=attributes,
            concat_features=True,
        )
    elif name == "embed":
        df1 = pd.read_csv("/vol/biomedic3/rrr2417/cxr-generation/src/mammo_artifacts/labelling_tools/manual_annotations_full_new.csv")
        df2 = pd.read_csv("/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv")
        data = df1.merge(df2, on="image_path")
        data["empi_anon"] = data["empi_anon_x"]

        MARKER_NAMES = ["circle marker", "triangle marker"]
        num_classes = len(MARKER_NAMES)
        data["multilabel_markers"] = data.apply(
            lambda row: np.array([row[name] for name in MARKER_NAMES]), axis=1
        )

        dm = EMBEDMammoDataModule(
            target="artifact_cancer_density",
            csv_file=data,
            # image_size=(256, 256),
            image_size=(192, 192),
            weighted_sampling=True,
            batch_alpha=0,
            batch_size=32,
            num_workers=1,
            return_dict=False,
            cache_size=32,
            normalise=True,
        )
        
        if split == "train":
            dataset = dm.train_set
        elif split == "valid":
            dataset = dm.val_set
        else:
            dataset = dm.test_set
    else:
        raise NotImplementedError(f'Dataset name `{name}` is not supported.')

    return dataset


class D2CCrop:
    """
    Almost same code as
        - https://github.com/phizaz/diffae/blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/dataset.py#L193-L217
    """
    def __init__(self):
        cx = 89
        cy = 121
        self.x1 = cy - 64
        self.x2 = cy + 64
        self.y1 = cx - 64
        self.y2 = cx + 64

    def __call__(self, img):
        img = torchvision.transforms.functional.crop(
            img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1,
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2}'


def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg['train']['dataset']
    else:
        transforms_cfg = cfg['test']['dataset']

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        elif t['name'] == 'D2CCrop':
            # For CerebA (not CelabA-HQ) dataset, D2C Crop is applied first.
            transform_cls = D2CCrop()
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_image_pillow(image_path):
    with Image.open(image_path) as img:
        image = img.convert('RGB')
    return image
