import json
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'


def featurize(batch, device):
    """Featurize single-chain structures with seq and coords (N, CA, C, O)."""
    B = len(batch)
    L_max = max(len(b['seq']) for b in batch)
    X = np.full((B, L_max, 4, 3), np.nan)
    residue_idx = np.full((B, L_max), -100, dtype=np.int64)
    chain_M = np.zeros((B, L_max), dtype=np.float32)
    chain_encoding_all = np.zeros((B, L_max), dtype=np.int64)
    S = np.zeros((B, L_max), dtype=np.int64)

    for i, b in enumerate(batch):
        length = len(b['seq'])
        X[i, :length] = np.stack([b['coords'][atom] for atom in ('N', 'CA', 'C', 'O')], axis=1)
        S[i, :length] = [alphabet.index(a) for a in b['seq']]
        residue_idx[i, :length] = np.arange(length)
        chain_M[i, :length] = 1
        chain_encoding_all[i, :length] = 1

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X = np.nan_to_num(X)

    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)

    return X, S, mask, chain_M, residue_idx, chain_encoding_all


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = F.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B, N, N, C] at Neighbor indices [B, N, K] => Neighbor features [B, N, K, C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B, N, C] at Neighbor indices [B, N, K] => [B, N, K, C]
    # Flatten and expand indices per batch [B, N, K] => [B, NK] => [B, NK, C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(neighbor_idx.shape[:3] + (-1, ))
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale

        self.act = torch.nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(DecLayer, self).__init__()
        self.scale = scale

        self.act = torch.nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset, mask):
        d = (offset + self.max_relative_feature).clip(
            0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 2)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self, edge_features, top_k=30, num_rbf=16, augment_eps=0., num_positional_embeddings=16
    ):
        """ Extract protein features """
        super().__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.augment_eps = augment_eps 

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A.unsqueeze(2) - B.unsqueeze(1)) ** 2, -1) + 1e-6) # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B.unsqueeze(-1), E_idx).squeeze(-1)           # [B, L, K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0: X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:, :, 1] - X[:, :, 0]
        c = X[:, :, 2] - X[:, :, 1]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1]
        Ca = X[:, :, 1]
        N = X[:, :, 0]
        C = X[:, :, 2]
        O = X[:, :, 3]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))          # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))      # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))      # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))      # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))    # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))     # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))     # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))     # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))    # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))      # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))      # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))     # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))     # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))     # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))      # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))     # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))     # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))     # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))    # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))      # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))      # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))     # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))     # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))     # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))      # C-O
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx.unsqueeze(2) - residue_idx.unsqueeze(1)    # [B, L, L]
        offset = gather_edges(offset.unsqueeze(-1), E_idx).squeeze(-1)  # [B, L, K]

        #find self vs non-self interaction
        d_chains = ((chain_labels.unsqueeze(2) - chain_labels.unsqueeze(1)) == 0).long()
        E_chains = gather_edges(d_chains.unsqueeze(-1), E_idx).squeeze(-1)
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(
        self, 
        num_letters=21, 
        edge_features=128,
        hidden_dim=128, 
        num_encoder_layers=3, 
        num_decoder_layers=3,
        vocab=21, 
        k_neighbors=32, 
        augment_eps=0.1, 
        dropout=0.1
    ):
        super().__init__()

        self.features = ProteinFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    
    def encode(self, X, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=X.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
        
        return h_V, h_E, E_idx
    

    def decode(self, S, h_V, h_E, E_idx, mask):
        device = S.device

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        mask_size = E_idx.shape[1]
        order_mask_backward = 1 - torch.triu(torch.ones(mask_size, mask_size, device=device))
        order_mask_backward = order_mask_backward.unsqueeze(0).expand(S.shape[0], -1, -1)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits[..., :-1], dim=-1)
        return log_probs

    def sample(self, h_V, h_E, E_idx, mask, temperature=1.0, eps=1e-8):
        device = h_V.device
        N_batch, N_nodes = h_V.shape[:2]

        mask_size = E_idx.shape[1]
        order_mask_backward = 1 - torch.triu(torch.ones(mask_size, mask_size, device=device))
        order_mask_backward = order_mask_backward.unsqueeze(0).expand(N_batch, -1, -1)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        h_S = torch.zeros_like(h_V, device=device)
        log_probs = torch.zeros((N_batch, N_nodes, 20), device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t in range(N_nodes):
            h_E_t = h_E[:, t: t + 1]
            mask_t = mask[:, t: t + 1]
            E_idx_t = E_idx[:, t: t + 1]
            h_EXV_encoder_t = h_EXV_encoder_fw[:, t: t + 1]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            for l, layer in enumerate(self.decoder_layers):
                # Updated relational features for future states
                h_V_t = h_V_stack[l][:, t: t + 1]
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_ESV_t = mask_bw[:, t: t + 1] * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l + 1][:, t] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
            # Sampling step
            h_V_t = h_V_stack[-1][:, t]
            logits = self.W_out(h_V_t)[:, :-1] / temperature

            prob_t = F.softmax(logits, dim=-1)
            log_probs[:, t] = (prob_t + eps).log()
            
            S_t = torch.multinomial(prob_t, 1).squeeze(-1)
            h_S[:, t] = self.W_s(S_t)
            S[:, t] = S_t
            
        return S, log_probs

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device=X.device

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M * mask # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device)))
        ) # Generate random decoding order: known positions (chain_M=0) first, then masked positions (chain_M=1)
        
        # Permutation matrix: maps original positions to decoding order
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
        
        # Apply permutation to lower triangular causal mask
        # order_mask_backward[b,q,p]=1 means position p is decoded before position q
        order_mask_backward = torch.einsum(
            'ij,biq,bjp->bqp', 
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))), 
            permutation_matrix_reverse, 
            permutation_matrix_reverse
        )

        # Gather attention mask for each position's K neighbors
        # mask_attend[b,i,k]=1 if neighbor k is decoded before position i
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

        # Backward mask: neighbors already decoded (can attend to)
        # Forward mask: neighbors not yet decoded (use encoder info instead)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class StructureDataset():
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file) as f:
            lines = f.readlines()
        for line in lines:
            self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
