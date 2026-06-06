from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter

import torch

from . import MAIN_METHODS
from .datasets import load_malimg_splits
from .encoders import build_imagenet_resnet18_encoder, build_malsim_resnet18_encoder, extract_features
from .metrics import macro_f1
from .methods import run_method
from .utils import load_json, log_info, resolve_device, set_seed, write_csv


def parse_methods(methods: list[str] | None) -> list[str]:
    selected = methods or list(MAIN_METHODS)
    invalid = [method for method in selected if method not in MAIN_METHODS]
    if invalid:
        raise ValueError(f"Supported methods are {MAIN_METHODS}; invalid={invalid}")
    return selected


def validate_task_plan(plan: dict[str, Any], class_names: list[str], shots: set[int] | None) -> list[dict[str, Any]]:
    if plan.get("class_names") != class_names:
        raise ValueError("Task-plan class names do not match the loaded Malimg train/val splits.")
    tasks = []
    for task in plan.get("tasks", []):
        shot = int(task["K"])
        if shots is not None and shot not in shots:
            continue
        if len(task["support_indices"]) != len(class_names) * shot:
            raise ValueError(f"Task repeat={task.get('repeat')} K={shot} does not contain exactly {shot} support samples per family.")
        label_counts = Counter(int(label) for label in task["support_labels"])
        expected_counts = {class_id: shot for class_id in range(len(class_names))}
        if dict(label_counts) != expected_counts:
            raise ValueError(f"Task repeat={task.get('repeat')} K={shot} has invalid per-family support counts.")
        tasks.append(task)
    if not tasks:
        raise ValueError("No tasks remain after applying the shot filter.")
    return tasks


def run_fewshot_main(
    *,
    task_plan: str | Path,
    malimg_train_dir: str | Path,
    malimg_val_dir: str | Path,
    malsim_checkpoint: str | Path,
    output_csv: str | Path,
    shots: list[int] | None,
    methods: list[str] | None,
    seed: int,
    batch_size: int,
    device_name: str,
    num_workers: int,
    image_size: int,
    deterministic: bool,
) -> None:
    set_seed(seed, deterministic=deterministic)
    device = resolve_device(device_name)
    selected_methods = parse_methods(methods)
    shot_filter = set(int(shot) for shot in shots) if shots else None

    log_info(f"Loading task plan: {task_plan}")
    plan = load_json(task_plan)
    train_ds, query_ds, class_names = load_malimg_splits(malimg_train_dir, malimg_val_dir, image_size=image_size)
    tasks = validate_task_plan(plan, class_names, shot_filter)

    log_info("Building encoders")
    visual_encoder = build_imagenet_resnet18_encoder()
    malware_encoder, checkpoint_info = build_malsim_resnet18_encoder(malsim_checkpoint)
    log_info(f"Loaded checkpoint: {checkpoint_info['checkpoint_sha256']}")

    visual_train, y_train = extract_features(visual_encoder, train_ds, batch_size, device, num_workers, seed, "ImgNet train")
    visual_query, y_query = extract_features(visual_encoder, query_ds, batch_size, device, num_workers, seed, "ImgNet val")
    malware_train, _ = extract_features(malware_encoder, train_ds, batch_size, device, num_workers, seed, "MalSim train")
    malware_query, _ = extract_features(malware_encoder, query_ds, batch_size, device, num_workers, seed, "MalSim val")
    features = {
        "visual_train": visual_train,
        "visual_query": visual_query,
        "malware_train": malware_train,
        "malware_query": malware_query,
    }
    labels = {"train": y_train, "query": y_query}

    rows = []
    for task in tasks:
        shot = int(task["K"])
        repeat = int(task["repeat"])
        query_indices = [int(i) for i in task["query_indices"]]
        y_true = y_query[query_indices].numpy()
        for method in selected_methods:
            log_info(f"Running method: {method}, K={shot}, repeat={repeat}")
            result = run_method(
                method,
                task,
                features,
                labels,
                train_ds,
                query_ds,
                len(class_names),
                device,
                str(malsim_checkpoint),
                batch_size,
                num_workers,
            )
            rows.append(
                {
                    "method": method,
                    "K": shot,
                    "repeat": repeat,
                    "macro_f1": macro_f1(y_true, result.predictions.numpy()),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(rows, output_csv)
    log_info(f"Saved raw results: {output_csv}")
