# vSmolLM

PyTorch implementation of GPT-2 from scratch with training on WikiText-103.

## Features

- Complete GPT-2 architecture implementation
- WikiText-103 dataset support with automatic download
- Mixed precision training (FP16), gradient accumulation, learning rate warmup
- Weights & Biases integration for experiment tracking

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login  # Optional
```

## Quick Start

Train a model:
```bash
python scripts/train.py --config configs/wikitext103.yaml
```

Evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoint_epoch_10.pt --split test
```

## Configuration

Edit `configs/wikitext103.yaml` to customize model architecture, training hyperparameters, and W&B settings.

## Project Structure

```
vSmolLM/
├── src/
│   ├── models/          # GPT-2 architecture
│   ├── data/            # Dataset loaders
│   ├── training/       # Training framework
│   └── config/          # Configuration management
├── scripts/             # Training and evaluation scripts
├── configs/             # Configuration files
└── outputs/             # Model checkpoints
```
