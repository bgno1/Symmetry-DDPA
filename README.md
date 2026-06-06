# Symmetric Dual-Domain Prototype Adaptation for Few-Shot Image-Based Malware Classification

This repository provides the core implementation for the manuscript **“Symmetric Dual-Domain Prototype Adaptation for Few-Shot Image-Based Malware Classification,”** which is currently under peer review.

## Dataset Preparation

Malimg should be arranged with existing train, validation, and test splits:

```text
malimg/
  train/
    family_1/
    ...
  val/
    family_1/
    ...
  test/
    family_1/
    ...
```

The main experiment draws support samples from `train` and evaluates query samples on `val`. The `test` split is not used in the main few-shot comparison.

Maldeb should be arranged as:

```text
maldeb/
  Benign/
  Malicious/
```

The `Benign` and `Malicious` folders are used only for image loading. These folder names are not used as supervised labels for SimCLR.

## Installation

```bash
pip install -r requirements.txt
```

## Running

Create the fixed task plan:

```bash
python scripts/01_make_task_plan.py \
  --malimg-train-dir /path/to/malimg/train \
  --malimg-val-dir /path/to/malimg/val \
  --output-json outputs/fixed_task_plan.json \
  --shots 1 5 10 20 \
  --repeats 20 \
  --seed 42
```

Pretrain the MalSim encoder:

```bash
python scripts/02_pretrain_malsim.py \
  --maldeb-dir /path/to/maldeb \
  --output-checkpoint checkpoints/malsim_resnet18_ep20_seed42.pth \
  --epochs 20 \
  --batch-size 256 \
  --lr 3e-4 \
  --image-size 128 \
  --seed 42 \
  --device cuda
```

Run the few-shot experiment:

```bash
python scripts/03_run_fewshot_main.py \
  --task-plan outputs/fixed_task_plan.json \
  --malimg-train-dir /path/to/malimg/train \
  --malimg-val-dir /path/to/malimg/val \
  --malsim-checkpoint checkpoints/malsim_resnet18_ep20_seed42.pth \
  --output-csv outputs/table1_fewshot_raw.csv \
  --shots 1 5 10 20 \
  --methods ImgNet-LP MalSim-LP ImgNet-FT MalSim-FT ImgNet-Proto MalSim-Proto DDPA \
  --seed 42 \
  --batch-size 128 \
  --device cuda
```
