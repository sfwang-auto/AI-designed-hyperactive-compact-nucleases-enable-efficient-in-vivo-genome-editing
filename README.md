# TnpBMPNN

TnpBMPNN is a ProteinMPNN-based model for structure-conditioned design of TnpB protein sequences.

This repository contains the code and pretrained models used in:

**AI-designed hyperactive compact nucleases enable efficient in vivo genome editing**

## Installation

Requires Python, NumPy, and a CUDA-enabled PyTorch installation.

Please install PyTorch according to your CUDA version:
https://pytorch.org/get-started/locally/

## Checkpoints

Three pretrained checkpoints are provided:

- `tnpb.pt`: TnpB-specific fine-tuned ProteinMPNN
- `round1.pt`: first-round activity-guided fine-tuned model
- `round2.pt`: second-round activity-guided fine-tuned model

## Sequence generation

Generate TnpB sequences from a backbone structure using:

```bash
python src/sample.py \
    --checkpoint checkpoints/round2.pt \
    --structure_path data/tnpb_wo_ctd.pkl \
    --path_for_outputs outputs \
    --n_samples 100 \
    --temperature 0.1
```
