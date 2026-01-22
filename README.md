# GPT-2 Training Repository

A complete PyTorch implementation for training GPT-2 models from scratch on various datasets, starting with WikiText-103. This repository includes a modular architecture, advanced training features, and Weights & Biases integration for experiment tracking.

## Features

- **GPT-2 Implementation from Scratch**: Complete PyTorch implementation of the GPT-2 architecture
- **WikiText-103 Support**: Built-in dataset loader with automatic download
- **Advanced Training Features**:
  - Gradient accumulation
  - Mixed precision training (FP16)
  - Learning rate scheduling with warmup
  - Gradient clipping
  - Model checkpointing
- **Weights & Biases Integration**: Comprehensive logging of metrics, hyperparameters, and model checkpoints
- **Evaluation Metrics**: Perplexity calculation on validation/test sets
- **Modular Design**: Easy to extend for other datasets and model variants

## Project Structure

```
GPT-scratch/
├── src/
│   ├── models/              # GPT-2 model architecture
│   │   ├── gpt2.py         # Main GPT-2 model
│   │   ├── attention.py    # Multi-head self-attention
│   │   ├── transformer_block.py  # Transformer decoder block
│   │   └── embeddings.py   # Token and position embeddings
│   ├── data/               # Data loading and processing
│   │   ├── dataset.py      # WikiText-103 dataset loader
│   │   ├── tokenizer.py    # GPT-2 tokenizer wrapper
│   │   └── collate.py      # Data collation functions
│   ├── training/           # Training framework
│   │   ├── trainer.py      # Main training loop
│   │   ├── evaluator.py    # Evaluation and metrics
│   │   └── utils.py        # Training utilities
│   └── config/             # Configuration management
│       └── config.py        # Config classes and loading
├── scripts/
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
├── configs/
│   └── wikitext103.yaml   # WikiText-103 configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GPT-scratch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up Weights & Biases:
```bash
wandb login
```

## Quick Start

### Training

Train a GPT-2 model on WikiText-103:

```bash
python scripts/train.py --config configs/wikitext103.yaml
```

The script will:
- Automatically download WikiText-103 dataset if not present
- Initialize the GPT-2 model
- Start training with W&B logging
- Save checkpoints at regular intervals

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoint_epoch_10.pt --split test
```

### Configuration

Edit `configs/wikitext103.yaml` to customize:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters (learning rate, batch size, etc.)
- W&B project settings
- Data loading settings

## Configuration Options

### Model Configuration

- `vocab_size`: Vocabulary size (default: 50257 for GPT-2)
- `embed_dim`: Embedding dimension (default: 768)
- `num_layers`: Number of transformer layers (default: 12)
- `num_heads`: Number of attention heads (default: 12)
- `ffn_dim`: Feed-forward network dimension (default: 3072)
- `max_seq_len`: Maximum sequence length (default: 1024)
- `dropout`: Dropout rate (default: 0.1)

### Training Configuration

- `batch_size`: Batch size per device
- `gradient_accumulation_steps`: Steps for gradient accumulation
- `learning_rate`: Learning rate (default: 6e-4)
- `max_epochs`: Maximum number of training epochs
- `warmup_steps`: Learning rate warmup steps
- `use_mixed_precision`: Enable FP16 training
- `eval_interval`: Steps between evaluations
- `save_interval`: Steps between checkpoint saves

### Weights & Biases

Configure W&B in the config file:
- `project`: W&B project name
- `entity`: W&B entity (username/team)
- `tags`: Tags for experiment organization
- `enabled`: Enable/disable W&B logging

## Model Architecture

The GPT-2 implementation includes:

1. **Embeddings**: Token and position embeddings with dropout
2. **Multi-Head Self-Attention**: Causal (masked) attention mechanism
3. **Transformer Blocks**: Pre-norm architecture with residual connections
4. **Feed-Forward Networks**: GELU-activated MLPs
5. **Layer Normalization**: Applied before attention and FFN
6. **Language Modeling Head**: Output projection to vocabulary

## Training Features

### Gradient Accumulation

Allows training with larger effective batch sizes when GPU memory is limited:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
```

### Mixed Precision Training

FP16 training for faster training and reduced memory usage:

```yaml
training:
  use_mixed_precision: true
```

### Learning Rate Scheduling

Cosine annealing with linear warmup:

```yaml
training:
  learning_rate: 6e-4
  warmup_steps: 2000
```

### Checkpointing

Models are automatically saved at regular intervals:
- Step-based checkpoints: `checkpoint_step_<step>.pt`
- Epoch-based checkpoints: `checkpoint_epoch_<epoch>.pt`
- Final model: `final_model.pt`

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/wikitext103.yaml --resume outputs/checkpoint_epoch_5.pt
```

## Evaluation Metrics

The evaluation script computes:
- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss), measures model uncertainty

## Extending to Other Datasets

To add support for other datasets:

1. Create a new dataset class in `src/data/dataset.py` (similar to `WikiText103Dataset`)
2. Create a configuration file in `configs/`
3. Update the training script to use your dataset

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_seq_len`
- Enable `use_mixed_precision: true`

### Slow Training

- Increase `num_workers` in data config
- Enable `pin_memory: true` for faster GPU transfer
- Use mixed precision training
- Ensure CUDA is available and properly configured

## License

[Add your license here]

## Acknowledgments

- OpenAI for the GPT-2 architecture
- HuggingFace for tokenizer utilities
- Weights & Biases for experiment tracking
