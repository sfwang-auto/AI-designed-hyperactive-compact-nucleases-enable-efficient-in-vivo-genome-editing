import os
import torch
import pickle
import random
import argparse
import numpy as np

from model_utils import alphabet, featurize, ProteinMPNN



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


def main(args):    
    set_seed()
    os.makedirs(args.path_for_outputs, exist_ok=True)
    device = torch.device('cuda:0')

    model = ProteinMPNN(k_neighbors=args.num_neighbors, augment_eps=0.)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    model.to(device)
    model.eval()
    
    with open(args.structure_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    pdb_dict = {
        'seq': pdb_dict['seq_chain_A'],
        'coords': {
            atom: pdb_dict['coords_chain_A'][f'{atom}_chain_A']
            for atom in ('N', 'CA', 'C', 'O')
        },
    }
    X, _, mask, _, residue_idx, chain_encoding_all = featurize([pdb_dict], device)
    
    with torch.inference_mode():
        h_V, h_E, E_idx = model.encode(X, mask, residue_idx, chain_encoding_all)
        S, _ = model.sample(
            h_V.expand(args.n_samples, -1, -1), 
            h_E.expand(args.n_samples, -1, -1, -1), 
            E_idx.expand(args.n_samples, -1, -1), 
            mask.expand(args.n_samples, -1), 
            temperature=args.temperature
        )
    
    fa = ''
    for i, S_i in enumerate(S):
        seq = ''.join([alphabet[idx.item()] for idx in S_i])
        fa += f'>{i}\n{seq}\n'
    
    fa_path = os.path.join(args.path_for_outputs, 'out')
    fa_path += f'_{args.temperature:.1f}.fa'
    with open(fa_path, 'w') as f:
        f.write(fa)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        '--path_for_outputs', type=str, 
        default='./output', 
        help='path for logs and model weights'
    )
    argparser.add_argument(
        '--structure_path', type=str, 
        default='./data/tnpb_wo_ctd.pkl', 
        help='path for loading data'
    ) 
    argparser.add_argument(
        '--checkpoint', type=str, help='path for previous model weights, e.g. file.pt'
    )
    argparser.add_argument(
        '--num_neighbors', type=int, default=48, help='number of neighbors for the sparse graph'
    )
    argparser.add_argument(
        '--n_samples', type=int, default=100, help='number of sequences for sampling'
    )
    argparser.add_argument(
        '--temperature', type=float, default=0.1, help='temperature for sampling'
    )
 
    args = argparser.parse_args()    
    main(args)   
