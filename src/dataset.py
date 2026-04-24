"""
dataset.py
==========
PyTorch Dataset and DataLoader utilities for the APTOS 2019
Diabetic Retinopathy detection challenge.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from preprocessing import (
    load_image,
    ben_graham_preprocess,
    clahe_lab_preprocess,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ─── Class metadata ─────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

NUM_CLASSES = 5


# ─── Albumentations transforms ───────────────────────────────────────────────

def get_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Heavy augmentation pipeline for training to improve generalisation.
    Includes spatial, photometric, and distortion transforms.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Deterministic transforms for validation/test (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─── Dataset class ───────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 retinal images.

    Parameters
    ----------
    csv_path     : str   Path to CSV file (columns: id_code, diagnosis)
    image_dir    : str   Directory containing .png images
    transform    : albumentations.Compose  Augmentation pipeline
    preprocess   : bool  Apply Ben Graham + CLAHE-LAB preprocessing before transforms
    img_size     : int   Target square image size
    """

    def __init__(self, csv_path: str, image_dir: str,
                 transform=None, preprocess: bool = True,
                 img_size: int = 512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.preprocess = preprocess
        self.img_size = img_size

        # Validate CSV columns
        if 'id_code' not in self.df.columns:
            raise ValueError(f"CSV '{csv_path}' must contain 'id_code' column")
        if 'diagnosis' not in self.df.columns:
            # Test set has no labels — use -1 as placeholder
            self.df['diagnosis'] = -1

        self.image_ids   = self.df['id_code'].values
        self.labels      = self.df['diagnosis'].values

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_preprocess(self, img_path: str) -> np.ndarray:
        """Load image and apply classical preprocessing if requested."""
        img = load_image(img_path, size=self.img_size)
        if self.preprocess:
            img = ben_graham_preprocess(img, sigmaX=10)
            img = clahe_lab_preprocess(img, clip_limit=2.0)
        return img  # uint8 HWC RGB

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        label  = int(self.labels[idx])
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        try:
            img = self._load_and_preprocess(img_path)
        except Exception as exc:
            # Return a black image on failure (prevents DataLoader crash)
            print(f"[APTOSDataset] Warning: failed to load '{img_path}': {exc}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label, img_id


# ─── DataLoader factory ──────────────────────────────────────────────────────

def build_dataloaders(
    data_path: str,
    batch_size: int = 16,
    img_size:   int = 512,
    num_workers: int = 4,
) -> tuple:
    """
    Build train and validation DataLoaders with WeightedRandomSampler for class imbalance.

    Parameters
    ----------
    data_path   : str   Root directory containing train.csv, valid.csv, train_images/, val_images/
    batch_size  : int   Samples per batch
    img_size    : int   Image size
    num_workers : int   Parallel data loading workers

    Returns
    -------
    tuple (train_loader, val_loader, class_weights)
    """
    train_csv   = os.path.join(data_path, 'train.csv')
    val_csv     = os.path.join(data_path, 'valid.csv')
    train_dir   = os.path.join(data_path, 'train_images')
    val_dir     = os.path.join(data_path, 'val_images')

    train_dataset = APTOSDataset(
        csv_path=train_csv, image_dir=train_dir,
        transform=get_train_transforms(img_size), preprocess=True, img_size=img_size
    )
    val_dataset = APTOSDataset(
        csv_path=val_csv, image_dir=val_dir,
        transform=get_val_transforms(img_size), preprocess=True, img_size=img_size
    )

    # ── WeightedRandomSampler to handle class imbalance ───────────────────
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, class_weights
