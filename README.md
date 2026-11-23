# JIT Micro Experiment

Train a JIT model on CelebA-HQ at 256 × 256.

## Setup

Python 3.12+ and CUDA-enabled PyTorch installed.
The scripts use project-relative cache defaults and accept an optional `.env` or `--env-file` for overrides.

## 1. Prepare data

```powershell
python prepare_data.py
```

This downloads CelebA-HQ, resizes the images to 256 × 256, and writes the train and validation split to `data/celebahq256-v1/`.

## 2. Train

```powershell
python train.py train --depth 2 --output runs/d2
```

The default recipe in `config.json` uses 80,000 updates at a constant learning rate of `8e-4` with zero warmup.
Use `--depth 4`, `8`, `12`, or `16` for a larger model and choose a fresh output directory for each run.

## 3. Evaluate and generate samples

```powershell
python eval.py --config runs/d2/resolved-config.json --checkpoint runs/d2/checkpoint-080000.pt --output runs/d2-eval --weights ema --sampler euler --steps 40
```

This writes held-out denoising scores and sample galleries conditioned on train and validation attributes.
Choose `--weights raw` for raw weights or `--sampler tweedie` for Tweedie sampling.

Share the Git source and keep local dotenv files, caches, run logs, and checkpoints private.
