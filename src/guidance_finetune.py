import os
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from openpyxl import load_workbook
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model_utils import featurize, ProteinMPNN


class Dataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, alphabet='ACDEFGHIKLMNPQRSTVWY'):
        super().__init__()
        self.labels = torch.tensor(labels)
        self.seqs = torch.tensor([[alphabet.index(aa) for aa in seq] for seq in seqs])

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def set_seed(seed=0):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sequence_scores(S, log_probs):
    return log_probs.gather(-1, S.unsqueeze(-1)).squeeze(-1).mean(-1)


def expand_features(features, mask, batch_size):
    return tuple(tensor.expand(batch_size, *tensor.shape[1:]) for tensor in (*features, mask))


def decode_sequences(model, S, features, mask):
    return model.decode(S, *expand_features(features, mask, S.size(0)))


def sample_sequences(model, features, mask, batch_size):
    S, _ = model.sample(*expand_features(features, mask, batch_size))
    return S


def kl_loss(log_probs, ref_log_probs, sampled_S):
    # Clone inference-mode samples so gather can save indices for backward.
    tokens = sampled_S.clone().unsqueeze(-1)
    log_p = log_probs.gather(-1, tokens).squeeze(-1)
    log_q = ref_log_probs.gather(-1, tokens).squeeze(-1)
    log_ratio = log_q - log_p
    return (log_ratio.exp() - log_ratio - 1).mean()


def loss_fn(S, log_probs, labels):
    B = S.size(0)
    log_probs = sequence_scores(S, log_probs)

    random_idx = torch.randperm(B)
    loss = (
        log_probs - log_probs[random_idx] - labels.log() + labels[random_idx].log()
    ).pow(2).mean()
    return loss


def main(args):
    set_seed()
    os.makedirs(args.path_for_outputs, exist_ok=True)
    device = torch.device('cuda:0')

    model = ProteinMPNN(k_neighbors=args.num_neighbors, augment_eps=0.)
    ref_model = ProteinMPNN(k_neighbors=args.num_neighbors, augment_eps=0.)
    model_state_dict = torch.load(args.ref_checkpoint, map_location='cpu')['model_state_dict']

    model.load_state_dict(model_state_dict)
    ref_model.load_state_dict(model_state_dict)

    model.to(device)
    ref_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open(args.structure_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    pdb_dict = {
        'seq': pdb_dict['seq_chain_A'],
        'coords': {atom: pdb_dict['coords_chain_A'][f'{atom}_chain_A'] for atom in ('N', 'CA', 'C', 'O')}
    }
    X, _, mask, _, residue_idx, chain_encoding_all = featurize([pdb_dict], device)

    seqs, labels = [], []
    wb = load_workbook(args.sequence_path)
    for row in wb[wb.sheetnames[0]].iter_rows(min_row=1, values_only=True):
        seqs.append(row[0])
        labels.append(row[1])
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        seqs, labels, test_size=0.2, random_state=0
    )

    train_dataset = Dataset(train_seqs, train_labels)
    val_dataset = Dataset(val_seqs, val_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

    ref_model.eval()
    # The reference model and structure stay fixed; cache the encoding once.
    with torch.inference_mode():
        ref_features = ref_model.encode(
            X, mask, residue_idx, chain_encoding_all,
        )
    best_val_loss = float('inf')
    checkpoint_name = os.path.splitext(os.path.basename(args.sequence_path))[0] + '.pt'
    best_checkpoint_path = os.path.join(args.path_for_outputs, checkpoint_name)
    for e in range(args.num_epochs):
        for S, labels in tqdm(train_loader, desc=f'training {e}/{args.num_epochs}'):
            optimizer.zero_grad()

            S = S.to(device)
            labels = labels.to(device)
            model.train()
            features = model.encode(X, mask, residue_idx, chain_encoding_all)
            log_probs = decode_sequences(model, S, features, mask)
            loss = loss_fn(S, log_probs, labels)
            # Backpropagate before changing modes so checkpoint recomputation uses training dropout.
            loss.backward()

            model.eval()
            kl_features = model.encode(X, mask, residue_idx, chain_encoding_all)
            with torch.inference_mode():
                sampled_S = sample_sequences(
                    model, kl_features, mask, args.kl_batch_size,
                )
                ref_log_probs = decode_sequences(
                    ref_model, sampled_S, ref_features, mask,
                )

            log_probs = decode_sequences(
                model, sampled_S.clone(), kl_features, mask,
            )
            kl = kl_loss(log_probs, ref_log_probs, sampled_S)

            # Keep eval mode during KL checkpoint recomputation; gradients still accumulate.
            (0.1 * kl).backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.inference_mode():
            features = model.encode(X, mask, residue_idx, chain_encoding_all)
            for S, labels in tqdm(val_loader, f'validation {e}/{args.num_epochs}'):
                S = S.to(device)

                log_probs = decode_sequences(model, S, features, mask)
                preds = sequence_scores(S, log_probs)

                val_labels.append(labels.cpu())
                val_preds.append(preds.cpu())

        residuals = torch.cat(val_preds) - torch.cat(val_labels).log()
        # Mean squared error over all pairs, matching expected random-pair training loss.
        val_loss = 2 * (residuals - residuals.mean()).square().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'num_edges' : args.num_neighbors,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_checkpoint_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        '--sequence_path', type=str, default='./data/round1.xlsx', help='path for fitness data'
    )
    argparser.add_argument(
        '--path_for_outputs', type=str, default='./outputs/guidance', help='path for model weights'
    )

    argparser.add_argument(
        '--structure_path', type=str, default='./data/tnpb_wo_ctd.pkl', help='path for structure data'
    )
    argparser.add_argument(
        '--ref_checkpoint', type=str, default='./checkpoints/tnpb.pt', help='path for reference model weights'
    )

    argparser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    argparser.add_argument('--num_epochs', type=int, default=20000, help='number of epochs')
    argparser.add_argument(
        '--kl_batch_size', type=int, default=32, help='number of sequences for kl of one batch'
    )
    argparser.add_argument(
        '--batch_size', type=int, default=64, help='number of sequences for one batch'
    )
    argparser.add_argument(
        '--num_neighbors', type=int, default=48, help='number of neighbors for the sparse graph'
    )

    args = argparser.parse_args()
    main(args)
