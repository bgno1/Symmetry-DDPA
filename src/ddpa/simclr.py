from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .datasets import MaldebSimCLRDataset
from .utils import ensure_dir, make_generator, save_json, seed_worker, sha256_file


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [
            torch.arange(batch, 2 * batch, device=z.device),
            torch.arange(0, batch, device=z.device),
        ]
    )
    return F.cross_entropy(logits, targets)


def pretrain_malsim(
    maldeb_dir: str | Path,
    output_checkpoint: str | Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    image_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    dataset = MaldebSimCLRDataset(maldeb_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    model = SimCLRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x1, x2 in tqdm(loader, desc=f"epoch {epoch}/{epochs}"):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses)})

    output_checkpoint = Path(output_checkpoint)
    ensure_dir(output_checkpoint.parent)
    config = {
        "maldeb_dir": str(Path(maldeb_dir).resolve()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature),
        "image_size": int(image_size),
        "seed": int(seed),
    }
    torch.save(
        {
            "backbone": model.encoder.state_dict(),
            "projection_head": model.projector.state_dict(),
            "simclr_config": config,
            "history": history,
        },
        output_checkpoint,
    )
    save_json(config, output_checkpoint.with_suffix(".config.json"))
    return {
        "checkpoint_path": str(output_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(output_checkpoint),
        "config_path": str(output_checkpoint.with_suffix(".config.json").resolve()),
        "final_loss": history[-1]["train_loss"] if history else None,
    }
