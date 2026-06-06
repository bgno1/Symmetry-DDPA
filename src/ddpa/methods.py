from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import models

from .encoders import load_malsim_backbone_state
from .utils import make_generator, seed_worker

MAIN_METHODS = [
    "ImgNet-LP",
    "MalSim-LP",
    "ImgNet-FT",
    "MalSim-FT",
    "ImgNet-Proto",
    "MalSim-Proto",
    "DDPA",
]

LP_DEFAULTS = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size_cap": 64,
}

FT_DEFAULTS = {
    "optimizer": "AdamW",
    "backbone_lr": 1e-5,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size_cap": 32,
}


@dataclass
class MethodResult:
    predictions: torch.Tensor
    metadata: dict[str, Any]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def build_class_prototypes(features: torch.Tensor, labels: torch.Tensor, support_indices: list[int], num_classes: int) -> torch.Tensor:
    idx = torch.as_tensor(support_indices, dtype=torch.long)
    support_x = features[idx]
    support_y = labels[idx]
    prototypes = []
    for class_id in range(num_classes):
        class_features = support_x[support_y == class_id]
        if class_features.numel() == 0:
            raise ValueError(f"No support samples for class {class_id}.")
        prototypes.append(l2_normalize(class_features.mean(dim=0, keepdim=True)).squeeze(0))
    return torch.stack(prototypes, dim=0)


def cosine_scores(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return l2_normalize(features) @ l2_normalize(prototypes).T


def predict_prototype(query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return cosine_scores(query_features, prototypes).argmax(dim=1)


def predict_ddpa(query_visual: torch.Tensor, visual_prototypes: torch.Tensor, query_malware: torch.Tensor, malware_prototypes: torch.Tensor) -> torch.Tensor:
    visual_scores = cosine_scores(query_visual, visual_prototypes)
    malware_scores = cosine_scores(query_malware, malware_prototypes)
    dual_scores = 0.5 * visual_scores + 0.5 * malware_scores
    return dual_scores.argmax(dim=1)


def train_linear_head(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> MethodResult:
    head = nn.Linear(support_x.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LP_DEFAULTS["lr"], weight_decay=LP_DEFAULTS["weight_decay"])
    loader = DataLoader(
        TensorDataset(support_x, support_y),
        batch_size=min(int(LP_DEFAULTS["batch_size_cap"]), len(support_y)),
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    for _ in range(int(LP_DEFAULTS["epochs"])):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        pred = head(query_x.to(device)).argmax(dim=1).cpu()
    return MethodResult(predictions=pred, metadata={"hyperparameters": LP_DEFAULTS})


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_finetune_model(method: str, num_classes: int, checkpoint_path: str | None) -> tuple[nn.Module, dict[str, Any]]:
    if method == "ImgNet-FT":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        metadata: dict[str, Any] = {}
    elif method == "MalSim-FT":
        if checkpoint_path is None:
            raise ValueError("MalSim-FT requires a MalSim checkpoint.")
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()
        metadata = load_malsim_backbone_state(model, checkpoint_path)
    else:
        raise ValueError(f"Unsupported fine-tuning method: {method}")
    model.fc = nn.Linear(512, num_classes)
    return model, metadata


def finetune_model(
    method: str,
    train_ds,
    query_ds,
    support_indices: list[int],
    query_indices: list[int],
    num_classes: int,
    device: torch.device,
    seed: int,
    checkpoint_path: str | None,
    batch_size: int,
    num_workers: int,
) -> MethodResult:
    model, metadata = build_finetune_model(method, num_classes, checkpoint_path)
    model = model.to(device)
    loader = DataLoader(
        Subset(train_ds, support_indices),
        batch_size=min(int(FT_DEFAULTS["batch_size_cap"]), len(support_indices)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    query_loader = DataLoader(Subset(query_ds, query_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (head_params if name.startswith("fc.") else backbone_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": FT_DEFAULTS["backbone_lr"]},
            {"params": head_params, "lr": FT_DEFAULTS["head_lr"]},
        ],
        weight_decay=FT_DEFAULTS["weight_decay"],
    )
    for _ in range(int(FT_DEFAULTS["epochs"])):
        model.train()
        freeze_batch_norm(model)
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in query_loader:
            preds.append(model(images.to(device, non_blocking=True)).argmax(dim=1).cpu())
    return MethodResult(predictions=torch.cat(preds, dim=0), metadata={"hyperparameters": FT_DEFAULTS, "checkpoint": metadata})


def run_method(
    method: str,
    task: dict[str, Any],
    features: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    train_ds,
    query_ds,
    num_classes: int,
    device: torch.device,
    checkpoint_path: str | None,
    batch_size: int,
    num_workers: int,
) -> MethodResult:
    if method not in MAIN_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    support = [int(i) for i in task["support_indices"]]
    query = [int(i) for i in task["query_indices"]]
    seed = int(task["seed"])
    y_support = labels["train"][support]

    visual_train = features["visual_train"]
    visual_query = features["visual_query"][query]
    malware_train = features["malware_train"]
    malware_query = features["malware_query"][query]

    if method == "ImgNet-Proto":
        prototypes = build_class_prototypes(visual_train, labels["train"], support, num_classes)
        return MethodResult(predict_prototype(visual_query, prototypes), {"feature_space": "ImageNet"})
    if method == "MalSim-Proto":
        prototypes = build_class_prototypes(malware_train, labels["train"], support, num_classes)
        return MethodResult(predict_prototype(malware_query, prototypes), {"feature_space": "MalSim"})
    if method == "DDPA":
        visual_prototypes = build_class_prototypes(visual_train, labels["train"], support, num_classes)
        malware_prototypes = build_class_prototypes(malware_train, labels["train"], support, num_classes)
        return MethodResult(predict_ddpa(visual_query, visual_prototypes, malware_query, malware_prototypes), {"fusion": "fixed_equal_weight"})
    if method == "ImgNet-LP":
        return train_linear_head(visual_train[support], y_support, visual_query, num_classes, device, seed)
    if method == "MalSim-LP":
        return train_linear_head(malware_train[support], y_support, malware_query, num_classes, device, seed)
    return finetune_model(method, train_ds, query_ds, support, query, num_classes, device, seed, checkpoint_path, batch_size, num_workers)
