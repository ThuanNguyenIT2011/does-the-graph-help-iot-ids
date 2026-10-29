"""Deletion/insertion curves for the saved IG (IG is not recomputed, no retraining).

For each explained flow, the 40 features are ranked by decreasing IG (support for the predicted class c first).
  - Deletion: replace the top m features with the baseline (0 = training mean) and measure p_c, m = 0..40.
  - Insertion: start from the baseline, restore the top m features to their true values and measure p_c.
Compared with random orders (N_RAND times). Area under the curve (AUC, x axis normalised to [0, 1]):
lower is better for deletion, higher is better for insertion.
DDoS is excluded as in the paper: the baseline falls in the DDoS region (p_DDoS(0) > 0.99999).
Graph and neighbours are held fixed, as in xai.py. Writes runs/xai_main_k10/faithfulness.json and .npz.
"""
import json, os, time
import numpy as np
import torch

from xgat import build, pick_device
from xai import RUN, OUT, SEEDS, BATCH, load_model

N_RAND = 5


@torch.no_grad()
def curve(model, x, nbr, ew, T, c, x0, order, insertion):
    """p_c after deleting (or inserting) the top m features in `order`, m = 0..F. Returns (|T|, F+1)."""
    F_ = x0.shape[1]
    out = []
    keep = torch.zeros_like(x0) if insertion else x0.clone()
    for m in range(F_ + 1):
        if m > 0:
            j = order[:, m - 1:m]
            keep.scatter_(1, j, x0.gather(1, j) if insertion else 0.0)
        x[T] = keep                                  # T are test nodes, none is a neighbour of another
        p = torch.softmax(model.forward_targets(x, nbr, ew, T), 1)
        out.append(p.gather(1, c).squeeze(1))
    x[T] = x0
    return torch.stack(out, 1)


def auc(p):
    return np.trapz(p, dx=1.0 / (p.shape[1] - 1), axis=1)


def main():
    device = pick_device()
    args = json.load(open(os.path.join(RUN, 'results.json')))['args']
    class A: pass
    a = A(); a.log = args['log']; a.k = args['k']
    G = build(a, device)
    x, nbr, ew = G['x'].to(device), G['nbr'].to(device), G['ew'].to(device)
    names = G['label_names']; y = G['y'].numpy(); F_ = x.shape[1]
    res, curves = {}, {}
    for seed in SEEDS:
        t0 = time.time()
        z = np.load(os.path.join(OUT, f'xai_seed{seed}.npz'))
        T_all = torch.tensor(z['T'], device=device); pred = torch.tensor(z['pred'], device=device)
        ig = torch.tensor(z['ig'], device=device)
        model = load_model(args, F_, len(names), seed, device)
        g = torch.Generator(device='cpu').manual_seed(seed)
        D, I, Dr, Ir = [], [], [], []
        for s in range(0, len(T_all), BATCH):
            T, c = T_all[s:s + BATCH], pred[s:s + BATCH].view(-1, 1)
            x0 = x[T].clone()
            order = ig[s:s + BATCH].argsort(1, descending=True)
            D.append(curve(model, x, nbr, ew, T, c, x0, order, False).cpu())
            I.append(curve(model, x, nbr, ew, T, c, x0, order, True).cpu())
            dr, ir = [], []
            for _ in range(N_RAND):
                r = torch.rand(len(T), F_, generator=g).argsort(1).to(device)
                dr.append(curve(model, x, nbr, ew, T, c, x0, r, False).cpu())
                ir.append(curve(model, x, nbr, ew, T, c, x0, r, True).cpu())
            Dr.append(torch.stack(dr).mean(0)); Ir.append(torch.stack(ir).mean(0))
        D, I, Dr, Ir = [torch.cat(v).numpy() for v in (D, I, Dr, Ir)]
        yt = y[z['T']]; P = z['pred']; ok = yt == P
        ddos = names.index('DDoS')
        groups = {'all_non_ddos': ok & (yt != ddos)}
        groups.update({n: ok & (yt == ci) for ci, n in enumerate(names) if ci != ddos})
        res[seed] = {}
        for n, m in groups.items():
            if not m.any(): continue
            res[seed][n] = dict(n=int(m.sum()),
                                del_ig=float(auc(D[m]).mean()), del_rand=float(auc(Dr[m]).mean()),
                                ins_ig=float(auc(I[m]).mean()), ins_rand=float(auc(Ir[m]).mean()))
        m = groups['all_non_ddos']
        curves[seed] = np.stack([D[m].mean(0), Dr[m].mean(0), I[m].mean(0), Ir[m].mean(0)])
        r = res[seed]['all_non_ddos']
        print(f"seed {seed}: {time.time() - t0:.0f}s | n {r['n']} | deletion IG {r['del_ig']:.3f} rand {r['del_rand']:.3f}"
              f" | insertion IG {r['ins_ig']:.3f} rand {r['ins_rand']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, 'faithfulness.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'faithfulness_curves.npz'), **{f'seed{s}': v for s, v in curves.items()},
             rows=np.array(['del_ig', 'del_rand', 'ins_ig', 'ins_rand']))


if __name__ == '__main__':
    main()
