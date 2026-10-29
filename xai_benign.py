"""IG cho logit margin với baseline = trung bình flow Benign của tập train
(không gian signed-log đã chuẩn hóa). Deletion/insertion thay đặc trưng bằng baseline này."""
import json, os, time
import numpy as np
import torch
from xgat import build, pick_device
from xai import RUN, OUT, SEEDS, BATCH, IG_STEPS, load_model
from faithfulness import auc, N_RAND
from xai_margin import margin

def ig_b(model, x, nbr, ew, T, c, b):
    x0 = x[T].clone(); d = x0 - b; total = torch.zeros_like(x0)
    for a in (torch.arange(IG_STEPS, device=x.device) + 0.5) / IG_STEPS:
        xs = x.clone(); inp = (b + a * d).requires_grad_(True); xs[T] = inp
        g, = torch.autograd.grad(margin(model.forward_targets(xs, nbr, ew, T), c).sum(), inp); total += g
    return (d * total / IG_STEPS).detach()

@torch.no_grad()
def curve_b(model, x, nbr, ew, T, c, x0, order, insertion, b):
    F_ = x0.shape[1]; out = []
    keep = b.expand_as(x0).clone() if insertion else x0.clone()
    for m in range(F_ + 1):
        if m > 0:
            j = order[:, m - 1:m]
            keep.scatter_(1, j, x0.gather(1, j) if insertion else b.expand_as(x0).gather(1, j))
        x[T] = keep
        out.append(torch.softmax(model.forward_targets(x, nbr, ew, T), 1).gather(1, c).squeeze(1))
    x[T] = x0
    return torch.stack(out, 1)

def main():
    device = pick_device()
    args = json.load(open(os.path.join(RUN, 'results.json')))['args']
    class A: pass
    a = A(); a.log = args['log']; a.k = args['k']
    G = build(a, device)
    x, nbr, ew = G['x'].to(device), G['nbr'].to(device), G['ew'].to(device)
    names, feat = G['label_names'], G['feat_cols']; y = G['y'].numpy(); F_ = x.shape[1]
    tr = G['masks']['train'].numpy(); ben = names.index('Benign')
    b = x[torch.tensor(tr[y[tr] == ben], device=device)].mean(0, keepdim=True)
    res = {}
    for seed in SEEDS:
        t0 = time.time()
        z = np.load(os.path.join(OUT, f'xai_seed{seed}.npz'))
        T_all = torch.tensor(z['T'], device=device); pred = torch.tensor(z['pred'], device=device)
        model = load_model(args, F_, len(names), seed, device)
        g = torch.Generator(device='cpu').manual_seed(seed)
        IG, M1, M0, D, I, Dr, Ir, P0 = [], [], [], [], [], [], [], []
        for s in range(0, len(T_all), BATCH):
            T, c = T_all[s:s + BATCH], pred[s:s + BATCH]; cc = c.view(-1, 1)
            ig = ig_b(model, x, nbr, ew, T, c, b)
            with torch.no_grad():
                m1 = margin(model.forward_targets(x, nbr, ew, T), c)
                xs = x.clone(); xs[T] = b.expand(len(T), -1); lg0 = model.forward_targets(xs, nbr, ew, T)
                m0 = margin(lg0, c); p0 = torch.softmax(lg0, 1).gather(1, cc).squeeze(1)
            x0 = x[T].clone(); order = ig.argsort(1, descending=True)
            D.append(curve_b(model, x, nbr, ew, T, cc, x0, order, False, b).cpu())
            I.append(curve_b(model, x, nbr, ew, T, cc, x0, order, True, b).cpu())
            dr, ir = [], []
            for _ in range(N_RAND):
                r = torch.rand(len(T), F_, generator=g).argsort(1).to(device)
                dr.append(curve_b(model, x, nbr, ew, T, cc, x0, r, False, b).cpu()); ir.append(curve_b(model, x, nbr, ew, T, cc, x0, r, True, b).cpu())
            Dr.append(torch.stack(dr).mean(0)); Ir.append(torch.stack(ir).mean(0))
            IG.append(ig.cpu()); M1.append(m1.cpu()); M0.append(m0.cpu()); P0.append(p0.cpu())
        IG, M1, M0, P0 = [torch.cat(v).numpy() for v in (IG, M1, M0, P0)]
        D, I, Dr, Ir = [torch.cat(v).numpy() for v in (D, I, Dr, Ir)]
        yt = y[z['T']]; P = z['pred']; ok = yt == P
        res[seed] = {}
        for ci, n in enumerate(names):
            mk = ok & (yt == ci)
            if not mk.any(): continue
            dm = M1[mk] - M0[mk]; gap = np.abs(IG[mk].sum(1) - dm) / np.maximum(np.abs(dm), 1e-12)
            res[seed][n] = dict(n=int(mk.sum()), p0_median=float(np.median(P0[mk])), dm_median=float(np.median(dm)),
                                gap_median=float(np.median(gap)), absig_median=float(np.median(np.abs(IG[mk]).sum(1))),
                                del_ig=float(auc(D[mk]).mean()), del_rand=float(auc(Dr[mk]).mean()),
                                ins_ig=float(auc(I[mk]).mean()), ins_rand=float(auc(Ir[mk]).mean()),
                                importance=dict(zip(feat, np.abs(IG[mk]).mean(0).tolist())))
        np.savez(os.path.join(OUT, f'xai_benignbase_seed{seed}.npz'), T=z['T'], ig=IG, m1=M1, m0=M0)
        d = res[seed]['DDoS']
        print(f"seed {seed}: {time.time()-t0:.0f}s | DDoS p0 {d['p0_median']:.3g} dm {d['dm_median']:.2f} gap {d['gap_median']:.4f} "
              f"sum|IG| {d['absig_median']:.2f} del {d['del_ig']:.3f}/{d['del_rand']:.3f} ins {d['ins_ig']:.3f}/{d['ins_rand']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, 'xai_benignbase.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
