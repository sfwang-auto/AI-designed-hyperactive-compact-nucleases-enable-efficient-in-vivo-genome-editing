import argparse
import os
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from model_utils import featurize, loss_smoothed, ProteinMPNN, StructureDataset


class ProteinMPNNModule(L.LightningModule):
    def __init__(self, checkpoint_path):
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.noise_level = checkpoint['noise_level']
        self.num_edges = checkpoint['num_edges']

        self.model = ProteinMPNN(k_neighbors=self.num_edges, augment_eps=self.noise_level)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        return self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

    def training_step(self, batch, batch_idx):
        _, S, mask, _, _, _ = batch
        log_probs = self(*batch)
        _, loss = loss_smoothed(S, log_probs, mask)
        return loss

    def validation_step(self, batch, batch_idx):
        _, S, mask, _, _, _ = batch
        log_probs = self(*batch)
        loss = F.nll_loss(log_probs.transpose(1, 2), S, reduction='none')
        return (loss * mask).sum() / mask.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class SaveBestModel(Callback):
    """Export the best weights using the single validation batch."""

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.best_loss = float('inf')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,):
        if trainer.sanity_checking or not trainer.is_global_zero: return

        val_loss = outputs.item()
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': pl_module.model.state_dict(),
                    'num_edges': pl_module.num_edges, 'noise_level': pl_module.noise_level
                }, os.path.join(self.output_dir, 'tnpb.pt')
            )


def main(args):
    collate_fn = partial(featurize, device=torch.device('cpu'))
    val_set = StructureDataset('./data/parsed_val_pdbs.jsonl')
    train_set = StructureDataset('./data/parsed_train_pdbs.jsonl')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    model = ProteinMPNNModule(args.previous_checkpoint)
    trainer = L.Trainer(
        devices=[0],
        logger=False,
        accelerator='gpu',
        max_epochs=args.num_epochs,
        enable_checkpointing=False,
        default_root_dir=args.path_for_outputs,
        callbacks=[SaveBestModel(args.path_for_outputs)]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--path_for_outputs", type=str, default="./outputs/family", help="path for model weights"
    )
    argparser.add_argument(
        "--previous_checkpoint", type=str, default="./checkpoints/v_48_020.pt",
        help="path for previous model weights"
    )
    argparser.add_argument("--num_epochs", type=int, default=2000, help="number of epochs to train for")
    argparser.add_argument("--batch_size", type=int, default=20, help="number of proteins for one batch")

    args = argparser.parse_args()
    main(args)
