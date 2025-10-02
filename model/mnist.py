"""
https://github.com/biomedia-mira/deepscm/blob/2dbfcde0d1fed63553c993d4abe4a74a59dc05d0/deepscm/datasets/morphomnist/__init__.py#L23
"""
import gzip
import os
import struct
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import VisionDataset


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack("BBBB", 0, 0, 0x08, data.ndim))
    f.write(struct.pack(">" + "I" * data.ndim, *data.shape))
    f.write(data.tobytes())


def save_idx(data: np.ndarray, path: str):
    """
    Writes an array to disk in IDX format.

    Parameters
    ----------
    data : array_like
        Input array of dtype ``uint8`` (will be coerced if different dtype).
    path : str
        Path of the output file. Will compress with `gzip` if path ends in '.gz'.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "wb") as f:
        _save_uint8(data, f)


def load_idx(path: str) -> np.ndarray:
    """
    Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path


def load_morphomnist_like(
    root_dir: str,
    train: bool = True,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)

    if columns is not None and "index" not in columns:
        usecols = ["index"] + list(columns)
    else:
        usecols = columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col="index")
    return images, labels, metrics


def save_morphomnist_like(
    images: np.ndarray,
    labels: np.ndarray,
    metrics: pd.DataFrame,
    root_dir, train: bool,
):
    """
    Args:
        images: array of MNIST-like images
        labels: array of class labels
        metrics: data frame of morphometrics
        root_dir: path to the target data directory
        train: whether to save as the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
    """
    assert len(images) == len(labels)
    assert len(images) == len(metrics)
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    os.makedirs(root_dir, exist_ok=True)
    save_idx(images, images_path)
    save_idx(labels, labels_path)
    metrics.to_csv(metrics_path, index_label="index")


def normalise(x, x_min, x_max):
    x = (x - x_min) / (x_max - x_min)  # [0,1]
    return 2 * x - 1  # [-1, 1]


def unnormalise(x, x_min, x_max):
    x = (x + 1) / 2  # [0, 1]
    return x * (x_max - x_min) + x_min


class MorphoMNIST(VisionDataset):

    min_max = {
        "slant": [-0.7, 0.7],
        "thickness": [1.27077, 7.15613],
        "intensity": [66.5, 254.0],
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        one_hot: bool = False,
        normalise_metadata: bool = False,
        features: List[str] = [],
        concat_features: bool = False,
    ):
        """
        Args:
            root_dir: path to data directory
            train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
                subset (``False``, ``'t10k-*'`` files)
            columns: list of morphometrics to load; by default (``None``) loads the image index and
                all available metrics: area, length, thickness, slant, width, and height
        """
        super().__init__(root, transform=transform)
        self.root_dir = root
        self.train = train
        self.images, self.labels, self.metrics_df = load_morphomnist_like(self.root_dir, self.train, None)
        self.features = features
        self.columns = self.metrics_df.columns
        self.normalise_metadata = normalise_metadata
        self.concat_features = concat_features

        self.metrics = {}
        for col in self.columns:
            if col not in self.features:
                continue
            print(col)
            if col in ["intensity", "thickness", "slant"] and self.normalise_metadata:
                self.metrics[col] = self.normalise_metadata_helper(col, self.metrics_df[col])
            else:
                self.metrics[col] = self.metrics_df[col]
        
        assert len(self.images) == len(self.labels) and len(self.images) == len(self.metrics_df)
        self.one_hot = one_hot

    @staticmethod
    def normalise_metadata_helper(col, values):
        return normalise(values, MorphoMNIST.min_max[col][0], MorphoMNIST.min_max[col][1])

    @staticmethod
    def unnormalise_metadata_helper(col, values):
        return unnormalise(values, MorphoMNIST.min_max[col][0], MorphoMNIST.min_max[col][1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.shape[0] == 3:
            img = np.moveaxis(img, 0, 2)
        img = Image.fromarray(img, mode="RGB" if "hue" in self.metrics else "L")
        if self.transform is not None:
            img = self.transform(img)
        metadata = {col: float(values[idx]) for col, values in self.metrics.items()}
        if "class" in self.features:
            metadata["class"] = torch.tensor(self.labels[idx]).long()
            if self.one_hot:
                metadata["class"] = F.one_hot(metadata["class"], num_classes=10)
        if self.concat_features:
            metadata["metadata"] = torch.cat([
                v if k == "class" else torch.tensor([v])
                for k, v in metadata.items() if k != "image"
            ])
        metadata["image"] = img
        return metadata["image"], metadata["metadata"].float()