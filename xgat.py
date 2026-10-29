"""XGAT: GATv2 over a cosine kNN graph in which every node links only to TRAINING nodes.

Differences from ADA-GNetIDS:
  - The graph is built without labels (the baseline uses val/test labels to choose anchors).
  - Val/test nodes link only to training nodes, never to each other.
  - Cosine similarity is used as an edge feature in attention (the baseline computes it but drops it).
  - GATv2 attention with residual connections, LayerNorm, jumping knowledge, and a branch that keeps the input features.
Since every node has exactly k neighbours, attention is computed densely as (N, k+1, d), which is fast on MPS.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_split, metrics

ROOT = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def knn_to_train(X_all, n_train, k, device, chunk=2048):
    """Return the indices of the k nearest training nodes (cosine) and their similarities for every node.
    A training node is never linked to itself."""
    Xn = F.normalize(torch.tensor(X_all, device=device), dim=1)
    bank = Xn[:n_train]
    N = Xn.shape[0]
    idx_out = torch.empty((N, k), dtype=torch.long)
    sim_out = torch.empty((N, k), dtype=torch.float32)
    for s in range(0, N, chunk):
        q = Xn[s:s + chunk]
        sim = q @ bank.T
        rows = torch.arange(s, min(s + chunk, N), device=device)
        is_train = rows < n_train
        if is_train.any():
            r = torch.nonzero(is_train).squeeze(1)
            sim[r, rows[r]] = -2.0  # exclude the node itself
        v, i = sim.topk(k, dim=1)
        idx_out[s:s + chunk] = i.cpu()
        sim_out[s:s + chunk] = v.cpu()
    return idx_out, sim_out


class DenseGATv2Layer(nn.Module):
    """GATv2 (Brody et al., 2022) over fixed k neighbours plus a self-loop, with an edge feature."""

    def __init__(self, d_in, d_out, heads=4, edge_dim=1, dropout=0.1):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.c = heads, d_out // heads
        self.lin_l = nn.Linear(d_in, d_out)
        self.lin_r = nn.Linear(d_in, d_out)
        self.lin_e = nn.Linear(edge_dim, d_out, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.c))
        nn.init.xavier_uniform_(self.att)
        self.dropout = dropout

    def forward(self, x, nbr, ew, return_att=False, x_dst=None, nbr_mask=None):
        # x: (N, d) source nodes; nbr: (M, k) indices into x; ew: (M, k, 1) similarities
        # x_dst: (M, d) target nodes (default x, when M = N); nbr_mask: (M, k) True = keep the edge (used for XAI)
        M, k = nbr.shape
        x_dst = x if x_dst is None else x_dst
        xl_all = self.lin_l(x).view(-1, self.h, self.c)        # sources (neighbours)
        xl_dst = self.lin_l(x_dst).view(M, self.h, self.c)
        xr = self.lin_r(x_dst).view(M, self.h, self.c)          # targets
        src = torch.cat([xl_dst.unsqueeze(1), xl_all[nbr]], dim=1)  # (M, k+1, h, c), position 0 is the node itself
        e_self = torch.ones(M, 1, ew.shape[-1], device=x.device)
        e = self.lin_e(torch.cat([e_self, ew], dim=1)).view(M, k + 1, self.h, self.c)
        z = F.leaky_relu(src + xr.unsqueeze(1) + e, 0.2)
        logit = (z * self.att).sum(-1)                           # (M, k+1, h)
        if nbr_mask is not None:
            keep = torch.cat([torch.ones(M, 1, dtype=torch.bool, device=x.device), nbr_mask], 1)
            logit = logit.masked_fill(~keep.unsqueeze(-1), float('-inf'))
        alpha = torch.softmax(logit, dim=1)
        alpha_d = F.dropout(alpha, self.dropout, self.training)
        out = (alpha_d.unsqueeze(-1) * src).sum(1).reshape(M, self.h * self.c)
        return (out, alpha) if return_att else out


class XGAT(nn.Module):
    def __init__(self, d_in, n_cls, hidden=128, layers=2, heads=4, dropout=0.2, self_only=False):
        super().__init__()
        self.self_only = self_only   # ablation: mask every edge to a neighbour, keep only the self-loop
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([DenseGATv2Layer(hidden, hidden, heads, 1, dropout) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden * (layers + 1), hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def forward(self, x, nbr, ew, return_att=False):
        h = self.inp(x)
        hs, atts = [h], []
        mask = torch.zeros_like(nbr, dtype=torch.bool) if self.self_only else None
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, nbr, ew, return_att, nbr_mask=mask)
            if return_att:
                m, a = m
                atts.append(a)
            h = norm(h + F.dropout(F.gelu(m), self.dropout, self.training))  # residual
            hs.append(h)
        logits = self.head(torch.cat(hs, dim=1))                         # jumping knowledge
        return (logits, atts) if return_att else logits

    def forward_targets(self, x, nbr, ew, T, nbr_mask_T=None, return_att=False):
        """Compute logits only for the target nodes T on their 2-hop subgraph (for XAI).
        The result equals the full-graph forward pass when no node of T is a neighbour of another
        (true for test nodes, which link only to training nodes).
        nbr_mask_T: (|T|, k) masks the edges of the target nodes themselves in both layers."""
        assert len(self.convs) == 2
        S1 = torch.unique(torch.cat([T, nbr[T].reshape(-1)]))
        S2 = torch.unique(torch.cat([S1, nbr[S1].reshape(-1)]))
        pos2 = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device); pos2[S2] = torch.arange(len(S2), device=x.device)
        pos1 = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device); pos1[S1] = torch.arange(len(S1), device=x.device)
        h0_2 = self.inp(x[S2])
        # layer 1 for S1
        m1_mask = None
        if nbr_mask_T is not None:
            m1_mask = torch.ones(len(S1), nbr.shape[1], dtype=torch.bool, device=x.device)
            m1_mask[pos1[T]] = nbr_mask_T
        h0_1 = h0_2[pos2[S1]]
        m = self.convs[0](h0_2, pos2[nbr[S1]], ew[S1], return_att, x_dst=h0_1, nbr_mask=m1_mask)
        if return_att: m, a1 = m
        h1_1 = self.norms[0](h0_1 + F.gelu(m))
        # layer 2 for T
        h1_T = h1_1[pos1[T]]
        m = self.convs[1](h1_1, pos1[nbr[T]], ew[T], return_att, x_dst=h1_T, nbr_mask=nbr_mask_T)
        if return_att: m, a2 = m
        h2_T = self.norms[1](h1_T + F.gelu(m))
        logits = self.head(torch.cat([h0_1[pos1[T]], h1_T, h2_T], 1))
        if return_att:
            return logits, (a1[pos1[T]], a2)
        return logits


def build(args, device):
    X, y, feat_cols, label_names = load_split(log_transform=args.log)
    X_all = np.vstack([X['train'], X['val'], X['test']])
    y_all = np.concatenate([y['train'], y['val'], y['test']])
    n_tr, n_va = len(y['train']), len(y['val'])
    t = time.time()
    nbr, sim = knn_to_train(X_all, n_tr, args.k, device)
    print(f'kNN: {X_all.shape[0]} nodes, k={args.k}, {time.time() - t:.1f}s', flush=True)
    masks = {
        'train': torch.arange(0, n_tr),
        'val': torch.arange(n_tr, n_tr + n_va),
        'test': torch.arange(n_tr + n_va, len(y_all)),
    }
    return dict(x=torch.tensor(X_all), y=torch.tensor(y_all, dtype=torch.long), nbr=nbr,
                ew=sim.unsqueeze(-1), masks=masks, feat_cols=feat_cols, label_names=label_names)


def train_one(G, args, seed, device, out_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    x, yv, nbr, ew = G['x'].to(device), G['y'].to(device), G['nbr'].to(device), G['ew'].to(device)
    m = {k: v.to(device) for k, v in G['masks'].items()}
    C = len(G['label_names'])
    counts = torch.bincount(yv[m['train']], minlength=C).float()
    cw = 1.0 / torch.sqrt(counts + 1e-6)
    cw = cw / cw.sum() * C                       # same class weighting as the baseline
    model = XGAT(x.shape[1], C, args.hidden, args.layers, args.heads, args.dropout,
                 self_only=args.ablate == 'selfloop').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=15)
    from sklearn.metrics import matthews_corrcoef
    best, best_ep, ckpt = -1.0, 0, os.path.join(out_dir, f'xgat_seed{seed}.pt')
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        logits = model(x, nbr, ew)
        loss = F.cross_entropy(logits[m['train']], yv[m['train']], weight=cw,
                               label_smoothing=args.ls)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(x, nbr, ew)[m['val']].argmax(1).cpu().numpy()
        val_mcc = matthews_corrcoef(yv[m['val']].cpu().numpy(), pv)   # model selection on the validation set
        sched.step(val_mcc)
        if val_mcc > best:
            best, best_ep = val_mcc, ep
            torch.save(model.state_dict(), ckpt)
        if ep - best_ep > args.patience:
            break
        if ep % 20 == 0:
            print(f'  seed {seed} ep {ep:03d} loss {loss.item():.4f} val_mcc {val_mcc:.4f} ({time.time() - t0:.0f}s)', flush=True)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(x, nbr, ew)
    prob = torch.softmax(logits[m['test']], 1).cpu().numpy()
    yt = yv[m['test']].cpu().numpy()
    res = metrics(yt, prob.argmax(1), prob, C)
    res.update(seed=seed, best_epoch=best_ep, best_val_mcc=best, train_s=time.time() - t0)
    np.savez(os.path.join(out_dir, f'preds_seed{seed}.npz'), y_true=yt, y_pred=prob.argmax(1), y_prob=prob)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--ls', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--log', type=int, default=1)
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--tag', type=str, default='dev')
    ap.add_argument('--ablate', choices=['none', 'selfloop', 'mlp'], default='none',
                    help="selfloop: attend only to the node itself; mlp: drop both GATv2 layers (encoder + head)")
    args = ap.parse_args()
    if args.ablate == 'mlp':
        args.layers = 0
    device = pick_device()
    out_dir = os.path.join(ROOT, 'runs', args.tag); os.makedirs(out_dir, exist_ok=True)
    print('device', device, vars(args), flush=True)
    G = build(args, device)
    rows = []
    for s in [int(v) for v in args.seeds.split(',')]:
        r = train_one(G, args, s, device, out_dir)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        rows.append(r)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'label_names': G['label_names'], 'runs': rows}, f, indent=1)


if __name__ == '__main__':
    main()
