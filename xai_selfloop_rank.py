"""IG (log p_c, baseline 0, 256 bước) cho model self-loop trên cùng 1.540 node,
so xếp hạng toàn cục với XGAT (Spearman)."""
import json, os, numpy as np, torch
from scipy.stats import spearmanr
from xgat import XGAT, build, pick_device
from xai import OUT, SEEDS, BATCH, IG_STEPS
ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_SL = os.path.join(ROOT, 'runs', 'ablate_selfloop_5seeds')
device = pick_device()
args = json.load(open(os.path.join(RUN_SL, 'results.json')))['args']
class A: pass
a = A(); a.log = args['log']; a.k = args['k']
G = build(a, device)
x, nbr, ew = G['x'].to(device), G['nbr'].to(device), G['ew'].to(device)
y = G['y'].numpy(); names = G['label_names']; dd = names.index('DDoS')
def ig_masked(m, x, nbr, ew, T, c, mask):
    x0 = x[T].clone(); total = torch.zeros_like(x0)
    for a in (torch.arange(IG_STEPS, device=x.device) + 0.5) / IG_STEPS:
        xs = x.clone(); inp = (a * x0).requires_grad_(True); xs[T] = inp
        lp = torch.log_softmax(m.forward_targets(xs, nbr, ew, T, nbr_mask_T=mask), 1)
        g, = torch.autograd.grad(lp.gather(1, c.view(-1, 1)).sum(), inp); total += g
    return (x0 * total / IG_STEPS).detach()
rho = []
for s in SEEDS:
    z = np.load(os.path.join(OUT, f'xai_seed{s}.npz')); T_all = torch.tensor(z['T'], device=device)
    m = XGAT(x.shape[1], len(names), args['hidden'], args['layers'], args['heads'], args['dropout'], self_only=True).to(device)
    m.load_state_dict(torch.load(os.path.join(RUN_SL, f'xgat_seed{s}.pt'), map_location=device, weights_only=True)); m.eval()
    IG, PR = [], []
    for i in range(0, len(T_all), BATCH):
        T = T_all[i:i + BATCH]
        mask = torch.zeros(len(T), nbr.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad(): pred = m.forward_targets(x, nbr, ew, T, nbr_mask_T=mask).argmax(1)
        with torch.no_grad():   # kiểm tra: forward có mask khớp forward self-only toàn đồ thị
            assert torch.allclose(m(x, nbr, ew)[T[:32]], m.forward_targets(x, nbr, ew, T[:32], nbr_mask_T=mask[:32]), atol=1e-3)
        IG.append(ig_masked(m, x, nbr, ew, T, pred, mask).cpu()); PR.append(pred.cpu())
    ig = torch.cat(IG).numpy()
    g_sl = np.abs(ig).mean(0); g_x = np.abs(z['ig']).mean(0)
    r = spearmanr(g_sl, g_x).correlation; rho.append(r)
    print(f'seed {s}: Spearman global |IG| self-loop vs XGAT = {r:.3f}', flush=True)
print('mean', np.mean(rho), 'min', np.min(rho))
