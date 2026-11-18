"""BDD100K 6-class driving scene classification dataset and data loaders."""

from __future__ import annotations

import json
import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

BDD_CLASSES: List[str] = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]

NUM_BDD_CLASSES: int = len(BDD_CLASSES)


def get_bdd_transforms(is_train: bool = True, img_size: int = 224) -> transforms.Compose:
    """Return train or eval preprocessing transforms for BDD100K driving scenes."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
    else:
        size = int((256 / 224) * img_size)
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])


class BDD100KClassificationDataset(Dataset):
    """Dataset for 6-class driving scene classification from BDD100K."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        image_dir: str,
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self._split_dir = None
        if split:
            cand = os.path.join(image_dir, "100k", split)
            if os.path.isdir(cand):
                self._split_dir = cand
            else:
                cand_10k = os.path.join(image_dir, "10k", split)
                if os.path.isdir(cand_10k):
                    self._split_dir = cand_10k

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_name: str) -> str:
        """Find the image path across flat image_dir or official 100k/10k subdirectories."""
        if self._split_dir:
            cand = os.path.join(self._split_dir, f"{image_name}.jpg")
            if os.path.isfile(cand):
                return cand
        direct = os.path.join(self.image_dir, f"{image_name}.jpg")
        if os.path.isfile(direct):
            return direct
        for split in ("train", "val", "test"):
            cand = os.path.join(self.image_dir, "100k", split, f"{image_name}.jpg")
            if os.path.isfile(cand):
                return cand
        for split in ("train", "val", "test"):
            cand = os.path.join(self.image_dir, "10k", split, f"{image_name}.jpg")
            if os.path.isfile(cand):
                return cand
        return direct

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_name, class_id = self.samples[idx]
        if self._split_dir:
            image_path = os.path.join(self._split_dir, f"{image_name}.jpg")
        else:
            image_path = self._resolve_image_path(image_name)

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
            image_path = self._resolve_image_path(image_name)
            with Image.open(image_path) as img:
                image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id


def load_bdd_split_samples(path: str) -> List[Tuple[str, int]]:
    """Parse BDD100K annotation JSON or TXT file returning (image_name, 0-indexed class_id)."""
    samples = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            img_name = s.get("image_name")
            class_id = s.get("class_id")
            if img_name is not None and class_id is not None:
                samples.append((img_name, int(class_id)))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_id = int(parts[1])
                    samples.append((image_name, class_id))
    return samples


def get_bdd_datasets(
    dataset_root: str = "data/bdd100k",
    seed: int = 42,
    img_size: int = 224,
    use_full: bool = False,
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    test_file: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
    """Return train, val, and test datasets for BDD100K driving scene classification."""
    annotations_dir = os.path.join(dataset_root, "annotations")
    image_dir = os.path.join(dataset_root, "images")

    if train_file:
        train_path = os.path.join(annotations_dir, train_file) if not os.path.isabs(train_file) else train_file
    elif use_full:
        train_path = os.path.join(annotations_dir, "train_full.json")
    else:
        train_path = os.path.join(annotations_dir, "train.json")

    if val_file:
        val_path = os.path.join(annotations_dir, val_file) if not os.path.isabs(val_file) else val_file
    elif use_full:
        val_path = os.path.join(annotations_dir, "val_full.json")
    else:
        val_path = os.path.join(annotations_dir, "val.json")

    if test_file:
        test_path = os.path.join(annotations_dir, test_file) if not os.path.isabs(test_file) else test_file
    elif use_full:
        test_path = os.path.join(annotations_dir, "test_full.json")
    else:
        test_path = os.path.join(annotations_dir, "test.json")

    # Fall back to .txt if .json does not exist
    if not os.path.isfile(train_path):
        train_path = os.path.join(annotations_dir, "train.txt")
    if not os.path.isfile(val_path):
        val_path = os.path.join(annotations_dir, "val.txt")
    if not os.path.isfile(test_path):
        test_path = os.path.join(annotations_dir, "test.txt")

    if not os.path.isfile(train_path) or not os.path.isfile(val_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"BDD100K annotation files missing in {annotations_dir}. "
            "Please run scripts/download_bdd100k.py first."
        )

    train_samples = load_bdd_split_samples(train_path)
    val_samples = load_bdd_split_samples(val_path)
    test_samples = load_bdd_split_samples(test_path)

    train_transform = get_bdd_transforms(is_train=True, img_size=img_size)
    eval_transform = get_bdd_transforms(is_train=False, img_size=img_size)

    train_dataset = BDD100KClassificationDataset(train_samples, image_dir, split="train", transform=train_transform)
    val_dataset = BDD100KClassificationDataset(val_samples, image_dir, split="val", transform=eval_transform)
    test_dataset = BDD100KClassificationDataset(test_samples, image_dir, split="test", transform=eval_transform)

    metadata = {
        "dataset": "BDD100K",
        "task": "driving_scene_classification",
        "classes": BDD_CLASSES,
        "num_classes": NUM_BDD_CLASSES,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "dataset_root": dataset_root,
    }

    return train_dataset, val_dataset, test_dataset, metadata
