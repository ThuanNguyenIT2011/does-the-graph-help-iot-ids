"""Completeness check of the saved IG (IG is not recomputed): for each explained flow,
f_c(x) and f_c(baseline) are the log-probability of the predicted class c when the features of the flow itself are x and 0
(graph and neighbours held fixed, as in xai.py). Gap = |ΣIG − (f_c(x) − f_c(0))|.
Writes runs/xai_main_k10/ig_completeness.json."""
import json, os
import numpy as np
import torch

from xgat import build, pick_device
from xai import RUN, OUT, SEEDS, BATCH, load_model


def main():
    device = pick_device()
    args = json.load(open(os.path.join(RUN, 'results.json')))['args']
    class A: pass
    a = A(); a.log = args['log']; a.k = args['k']
    G = build(a, device)
    x, nbr, ew = G['x'].to(device), G['nbr'].to(device), G['ew'].to(device)
    names = G['label_names']; y = G['y'].numpy()
    res = {}
    for seed in SEEDS:
        z = np.load(os.path.join(OUT, f'xai_seed{seed}.npz'))
        T_all = torch.tensor(z['T'], device=device); pred = torch.tensor(z['pred'], device=device)
        model = load_model(args, x.shape[1], len(names), seed, device)
        fx, f0, pb = [], [], []
        with torch.no_grad():
            for s in range(0, len(T_all), BATCH):
                T, c = T_all[s:s + BATCH], pred[s:s + BATCH].view(-1, 1)
                lp = torch.log_softmax(model.forward_targets(x, nbr, ew, T), 1)
                xs = x.clone(); xs[T] = 0
                lp0 = torch.log_softmax(model.forward_targets(xs, nbr, ew, T), 1)
                fx.append(lp.gather(1, c).squeeze(1).cpu()); f0.append(lp0.gather(1, c).squeeze(1).cpu())
                pb.append(lp0.exp().cpu())
        fx, f0, pb = torch.cat(fx).numpy(), torch.cat(f0).numpy(), torch.cat(pb).numpy()
        S = z['ig'].sum(1); df = fx - f0; gap = np.abs(S - df)
        yt = y[z['T']]; ok = yt == z['pred']
        per = {}
        for ci, n in enumerate(names):
            m = ok & (yt == ci)
            if not m.any(): continue
            per[n] = dict(n=int(m.sum()), median_delta_f=float(np.median(df[m])), median_sumIG=float(np.median(S[m])),
                          median_gap=float(np.median(gap[m])), median_rel_gap=float(np.median(gap[m] / np.maximum(np.abs(df[m]), 1e-12))),
                          median_p_c_baseline=float(np.median(np.exp(f0[m]))))
        res[seed] = dict(baseline_pred_share={names[i]: float((pb.argmax(1) == i).mean()) for i in range(len(names))},
                         median_p_ddos_baseline=float(np.median(pb[:, names.index('DDoS')])),
                         min_p_ddos_baseline=float(pb[:, names.index('DDoS')].min()),
                         per_class=per)
        print(seed, json.dumps(res[seed], indent=0)[:1500], flush=True)
    json.dump(res, open(os.path.join(OUT, 'ig_completeness.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
