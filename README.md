# JIT Micro Experiment

Train a JIT model on CelebA-HQ at 256 × 256.

[X post](https://x.com/broq_dev/status/2124765551516176512)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the project directory.

### Windows (NVIDIA CUDA)

```powershell
uv venv --python 3.12
uv pip install --torch-backend=auto "torch==2.13.0" -e ".[windows-compile]"
. ./.venv/Scripts/Activate.ps1
```

### macOS (Apple silicon, macOS 14+)

```bash
uv venv --python 3.12
uv pip install "torch==2.13.0" -e .
source .venv/bin/activate
```

On macOS, set `runtime.device` to `"cpu"` and set `runtime.compile` and `runtime.cuda_graph` to `false` in `config.json`.
Apple GPU acceleration via MPS is unsupported.

### Linux (NVIDIA CUDA)

```bash
uv venv --python 3.12
uv pip install --torch-backend=auto "torch==2.13.0" -e .
source .venv/bin/activate
```

For CPU-only Linux, use the macOS config settings above.
The scripts use project-relative cache defaults and accept an optional `.env` or `--env-file` for overrides.

## 1. Prepare data

```sh
python prepare_data.py
```

This downloads CelebA-HQ, resizes the images to 256 × 256, and writes the train and validation split to `data/celebahq256-v1/`.

## 2. Train

```sh
python train.py train --depth 2 --output runs/d2
```

The default recipe in `config.json` uses 80,000 updates at a constant learning rate of `8e-4` with zero warmup.
Use `--depth 4`, `8`, `12`, or `16` for a larger model and choose a fresh output directory for each run.

## 3. Evaluate and generate samples

```sh
python eval.py --config runs/d2/resolved-config.json --checkpoint runs/d2/checkpoint-080000.pt --output runs/d2-eval --weights ema --sampler euler --steps 40
```

This writes held-out denoising scores and sample galleries conditioned on train and validation attributes.
Choose `--weights raw` for raw weights or `--sampler tweedie` for Tweedie sampling.

Share the Git source and keep local dotenv files, caches, run logs, and checkpoints private.

## References

- Mariia Drozdova, Stéphane Liem Nguyen, and François Fleuret (2026). [Let It Go or Learn to Self-Correct: Continuous Diffusion for Constrained Discrete Tasks](https://arxiv.org/abs/2609.09009).
- Donghoon Ahn et al. (2024). [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377).
