# TnpBMPNN

TnpBMPNN is a ProteinMPNN-based model for structure-conditioned design of TnpB protein sequences.

This repository contains the code and pretrained models used in:

**AI-designed hyperactive compact nucleases enable efficient in vivo genome editing**

## Installation

Requires Python, NumPy, and a CUDA-enabled PyTorch installation.

Please install PyTorch according to your CUDA version:
https://pytorch.org/get-started/locally/

For fine-tuning, install the additional dependencies:

```bash
pip install numpy lightning tqdm openpyxl scikit-learn
```

Run the commands below from the repository root. Both fine-tuning scripts require a CUDA GPU.

## Data

Download the five data files from [Zenodo](https://zenodo.org/records/24321081) and place them in `data/`.

## Checkpoints

Three pretrained checkpoints are provided:

- `tnpb.pt`: TnpB-specific fine-tuned ProteinMPNN
- `round1.pt`: first-round activity-guided fine-tuned model
- `round2.pt`: second-round activity-guided fine-tuned model

## Fine-tuning

### Family-specific fine-tuning

Uses `data/parsed_train_pdbs.jsonl` and `data/parsed_val_pdbs.jsonl`.

```bash
python src/family_finetune.py
```

Saves the best model to `outputs/family/tnpb.pt`.

### Activity-guided fine-tuning

```bash
python src/guidance_finetune.py --sequence_path data/round1.xlsx
python src/guidance_finetune.py --sequence_path data/round2.xlsx
```

Outputs `outputs/guidance/round1.pt` and `outputs/guidance/round2.pt`, respectively.

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
