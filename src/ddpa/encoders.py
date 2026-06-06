from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .utils import make_generator, seed_worker, sha256_file


class ResNet18Encoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        if features.shape[1] != 512:
            raise RuntimeError(f"Expected 512-dimensional features, got {features.shape[1]}.")
        return features


def _feature_backbone(weights: Any) -> nn.Module:
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def build_imagenet_resnet18_encoder() -> ResNet18Encoder:
    encoder = ResNet18Encoder(_feature_backbone(models.ResNet18_Weights.IMAGENET1K_V1))
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _clean_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ("module.", "backbone.", "encoder."):
            new_key = new_key.removeprefix(prefix)
        cleaned[new_key] = value
    return cleaned


def _extract_state(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state-dict-like mapping.")
    for key in ("backbone", "encoder_state_dict", "encoder", "model_state_dict", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return _clean_state_dict(checkpoint[key]), key
    return _clean_state_dict(checkpoint), "raw_state_dict"


def load_malsim_backbone_state(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MalSim checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, format_key = _extract_state(checkpoint)
    if not state:
        raise ValueError("No tensor state was found in the MalSim checkpoint.")
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint keys do not match the ResNet-18 encoder. missing={missing}, unexpected={unexpected}")
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format_key": format_key,
        "loaded_tensor_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def build_malsim_resnet18_encoder(checkpoint_path: str | Path) -> tuple[ResNet18Encoder, dict[str, Any]]:
    model = _feature_backbone(weights=None)
    metadata = load_malsim_backbone_state(model, checkpoint_path)
    encoder = ResNet18Encoder(model)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, metadata


def extract_features(
    encoder: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    desc: str,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    features, labels = [], []
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        for images, y in tqdm(loader, desc=desc):
            z = encoder(images.to(device, non_blocking=True))
            if normalize:
                z = F.normalize(z, p=2, dim=1)
            features.append(z.cpu())
            labels.append(y.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)
