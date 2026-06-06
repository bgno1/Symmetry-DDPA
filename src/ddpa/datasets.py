from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from .utils import save_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_SHOTS = [1, 5, 10, 20]


def pil_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def eval_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def simclr_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TwoViewTransform:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


class RemappedImageFolder(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageFolder, indices: list[int], label_map: dict[int, int], class_names: list[str]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        self.classes = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.targets = [label_map[int(dataset.targets[i])] for i in indices]
        self.samples = [(dataset.samples[i][0], label_map[int(dataset.samples[i][1])]) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[self.indices[index]]
        return image, self.label_map[int(label)]


class MaldebSimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, image_size: int) -> None:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Maldeb directory does not exist: {root}")
        self.dataset = datasets.ImageFolder(root=root, transform=TwoViewTransform(simclr_transform(image_size)), loader=pil_rgb)
        class_names = set(self.dataset.classes)
        if class_names != {"Benign", "Malicious"}:
            raise ValueError(f"Expected Maldeb folders Benign and Malicious, found {sorted(class_names)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        (x1, x2), _ = self.dataset[index]
        return x1, x2


def imagefolder(root: str | Path, transform: Callable | None = None) -> datasets.ImageFolder:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    return datasets.ImageFolder(root=root, transform=transform, loader=pil_rgb)


def load_malimg_splits(train_dir: str | Path, val_dir: str | Path, image_size: int = 128) -> tuple[RemappedImageFolder, RemappedImageFolder, list[str]]:
    transform = eval_transform(image_size)
    train = imagefolder(train_dir, transform)
    val = imagefolder(val_dir, transform)
    if set(train.classes) != set(val.classes):
        raise ValueError(f"Train/val family names differ: train={sorted(train.classes)}, val={sorted(val.classes)}")
    if len(train.classes) != 25:
        raise ValueError(f"Expected 25 Malimg families, found {len(train.classes)}")
    class_names = list(train.classes)
    final_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_map = {train.class_to_idx[name]: final_to_idx[name] for name in class_names}
    val_map = {val.class_to_idx[name]: final_to_idx[name] for name in val.classes}
    train_ds = RemappedImageFolder(train, list(range(len(train))), train_map, class_names)
    val_ds = RemappedImageFolder(val, list(range(len(val))), val_map, class_names)
    return train_ds, val_ds, class_names


def class_counts(dataset: RemappedImageFolder) -> dict[str, int]:
    counts = {name: 0 for name in dataset.classes}
    for label in dataset.targets:
        counts[dataset.classes[int(label)]] += 1
    return counts


def sample_support_indices(dataset: RemappedImageFolder, shot: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        by_class[int(label)].append(index)
    if sorted(by_class) != list(range(len(dataset.classes))):
        raise ValueError("Training split does not contain all expected families.")
    indices: list[int] = []
    for class_id in sorted(by_class):
        pool = by_class[class_id]
        if len(pool) < shot:
            raise ValueError(f"Family {dataset.classes[class_id]} has {len(pool)} train samples, need {shot}.")
        indices.extend(int(i) for i in rng.choice(pool, size=shot, replace=False))
    return indices


def build_fixed_task_plan(
    train_ds: RemappedImageFolder,
    query_ds: RemappedImageFolder,
    train_dir: str | Path,
    val_dir: str | Path,
    shots: list[int],
    repeats: int,
    seed: int,
) -> dict:
    if any(int(shot) not in EXPECTED_SHOTS for shot in shots):
        raise ValueError(f"Supported shots are {EXPECTED_SHOTS}.")
    tasks = []
    for shot in shots:
        for repeat in range(repeats):
            task_seed = int(seed) + int(shot) * 1000 + int(repeat)
            support_indices = sample_support_indices(train_ds, int(shot), task_seed)
            tasks.append(
                {
                    "K": int(shot),
                    "repeat": int(repeat),
                    "seed": int(task_seed),
                    "class_names": list(train_ds.classes),
                    "support_indices": support_indices,
                    "support_paths": [str(Path(train_ds.samples[i][0]).resolve()) for i in support_indices],
                    "support_labels": [int(train_ds.targets[i]) for i in support_indices],
                    "query_indices": list(range(len(query_ds))),
                    "query_paths": [str(Path(path).resolve()) for path, _ in query_ds.samples],
                    "query_labels": [int(label) for label in query_ds.targets],
                }
            )
    return {
        "protocol": "Malimg train-support and validation-query few-shot evaluation",
        "seed": int(seed),
        "repeats": int(repeats),
        "shots": [int(shot) for shot in shots],
        "class_names": list(train_ds.classes),
        "train_split_root": str(Path(train_dir).resolve()),
        "validation_split_root": str(Path(val_dir).resolve()),
        "train_class_counts": class_counts(train_ds),
        "validation_class_counts": class_counts(query_ds),
        "tasks": tasks,
    }


def save_task_plan(plan: dict, output_json: str | Path) -> None:
    save_json(plan, output_json)
