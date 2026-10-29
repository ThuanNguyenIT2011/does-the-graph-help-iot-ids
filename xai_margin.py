"""IG of the logit margin m_c = z_c - max_{c'!=c} z_{c'} instead of log p_c, zero baseline as before.
log p_c saturates for DDoS (p_c(0) > 0.99999), so IG is close to 0; the margin does not saturate.
Measures: completeness (sum IG against m(x) - m(0)), sum |IG| per class, feature ranking for DDoS,
deletion/insertion in IG-margin order (p_c, as in faithfulness.py). Reuses the explained nodes (xai_seed*.npz)."""
import json, os, time
import numpy as np
import torch
from xgat import build, pick_device
from xai import RUN, OUT, SEEDS, BATCH, IG_STEPS, load_model
from faithfulness import curve, auc, N_RAND

def margin(logits, c):
    zc = logits.gather(1, c.view(-1, 1)).squeeze(1)
    other = logits.scatter(1, c.view(-1, 1), float('-inf')).max(1).values
    return zc - other

def ig_margin(model, x, nbr, ew, T, c):
    x0 = x[T].clone(); total = torch.zeros_like(x0)
    for a in (torch.arange(IG_STEPS, device=x.device) + 0.5) / IG_STEPS:
        xs = x.clone(); inp = (a * x0).requires_grad_(True); xs[T] = inp
        m = margin(model.forward_targets(xs, nbr, ew, T), c).sum()
        g, = torch.autograd.grad(m, inp); total += g
    return (x0 * total / IG_STEPS).detach()

def main():
    device = pick_device()
    args = json.load(open(os.path.join(RUN, 'results.json')))['args']
    class A: pass
    a = A(); a.log = args['log']; a.k = args['k']
    G = build(a, device)
    x, nbr, ew = G['x'].to(device), G['nbr'].to(device), G['ew'].to(device)
    names, feat = G['label_names'], G['feat_cols']; y = G['y'].numpy(); F_ = x.shape[1]
    res = {}
    for seed in SEEDS:
        t0 = time.time()
        z = np.load(os.path.join(OUT, f'xai_seed{seed}.npz'))
        T_all = torch.tensor(z['T'], device=device); pred = torch.tensor(z['pred'], device=device)
        model = load_model(args, F_, len(names), seed, device)
        g = torch.Generator(device='cpu').manual_seed(seed)
        IG, M1, M0, D, I, Dr, Ir = [], [], [], [], [], [], []
        for s in range(0, len(T_all), BATCH):
            T, c = T_all[s:s + BATCH], pred[s:s + BATCH]
            ig = ig_margin(model, x, nbr, ew, T, c)
            with torch.no_grad():
                m1 = margin(model.forward_targets(x, nbr, ew, T), c)
                xs = x.clone(); xs[T] = 0.0; m0 = margin(model.forward_targets(xs, nbr, ew, T), c)
            x0 = x[T].clone(); order = ig.argsort(1, descending=True); cc = c.view(-1, 1)
            D.append(curve(model, x, nbr, ew, T, cc, x0, order, False).cpu())
            I.append(curve(model, x, nbr, ew, T, cc, x0, order, True).cpu())
            dr, ir = [], []
            for _ in range(N_RAND):
                r = torch.rand(len(T), F_, generator=g).argsort(1).to(device)
                dr.append(curve(model, x, nbr, ew, T, cc, x0, r, False).cpu()); ir.append(curve(model, x, nbr, ew, T, cc, x0, r, True).cpu())
            Dr.append(torch.stack(dr).mean(0)); Ir.append(torch.stack(ir).mean(0))
            IG.append(ig.cpu()); M1.append(m1.cpu()); M0.append(m0.cpu())
        IG, M1, M0 = [torch.cat(v).numpy() for v in (IG, M1, M0)]
        D, I, Dr, Ir = [torch.cat(v).numpy() for v in (D, I, Dr, Ir)]
        yt = y[z['T']]; P = z['pred']; ok = yt == P
        res[seed] = {}
        for ci, n in enumerate(names):
            mk = ok & (yt == ci)
            if not mk.any(): continue
            dm = M1[mk] - M0[mk]; gap = np.abs(IG[mk].sum(1) - dm) / np.maximum(np.abs(dm), 1e-12)
            res[seed][n] = dict(n=int(mk.sum()), m0_median=float(np.median(M0[mk])), dm_median=float(np.median(dm)),
                                gap_median=float(np.median(gap)), absig_median=float(np.median(np.abs(IG[mk]).sum(1))),
                                del_ig=float(auc(D[mk]).mean()), del_rand=float(auc(Dr[mk]).mean()),
                                ins_ig=float(auc(I[mk]).mean()), ins_rand=float(auc(Ir[mk]).mean()),
                                top_features=[feat[j] for j in np.argsort(-np.abs(IG[mk]).mean(0))[:8]],
                                importance=dict(zip(feat, np.abs(IG[mk]).mean(0).tolist())))
        np.savez(os.path.join(OUT, f'xai_margin_seed{seed}.npz'), T=z['T'], ig=IG, m1=M1, m0=M0)
        d = res[seed]['DDoS']
        print(f"seed {seed}: {time.time()-t0:.0f}s | DDoS n {d['n']} m0 {d['m0_median']:.2f} dm {d['dm_median']:.2f} gap {d['gap_median']:.4f} "
              f"sum|IG| {d['absig_median']:.2f} del {d['del_ig']:.3f}/{d['del_rand']:.3f} ins {d['ins_ig']:.3f}/{d['ins_rand']:.3f} top {d['top_features'][:5]}", flush=True)
    json.dump(res, open(os.path.join(OUT, 'xai_margin.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
