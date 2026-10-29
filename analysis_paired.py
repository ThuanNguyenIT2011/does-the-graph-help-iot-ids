"""Analysis of saved results only, no retraining.
Paired per-seed differences and bootstrap CIs on the test set.
Mean and median |IG| of the top-ranked features.
"""
import json, os, numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score
from common import load_split
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
runs = {'XGAT': 'main_k10_5seeds', 'Self-loop': 'ablate_selfloop_5seeds', 'MLP': 'ablate_mlp_5seeds'}
P = {m: [np.load(f'{R}/{t}/preds_seed{s}.npz') for s in range(5)] for m, t in runs.items()}
yt = P['XGAT'][0]['y_true']
rng = np.random.default_rng(0); B = 1000
idx = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
def stat(yp, i): return matthews_corrcoef(yt[i], yp[i]), f1_score(yt[i], yp[i], average='macro')
for a, b in [('Self-loop', 'XGAT'), ('XGAT', 'MLP')]:
    d = np.array([[matthews_corrcoef(yt, P[a][s]['y_pred']) - matthews_corrcoef(yt, P[b][s]['y_pred']),
                   f1_score(yt, P[a][s]['y_pred'], average='macro') - f1_score(yt, P[b][s]['y_pred'], average='macro')] for s in range(5)])
    # bootstrap over test flows: each resample redraws the test set; differences averaged over the 5 seeds
    bs = []
    for i in idx:
        bs.append(np.mean([np.subtract(stat(P[a][s]['y_pred'], i), stat(P[b][s]['y_pred'], i)) for s in range(5)], 0))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    print(f'{a} - {b}: paired MCC {d[:,0].mean():+.4f}±{d[:,0].std(ddof=1):.4f} per-seed {np.round(d[:,0],4)} | boot95 [{lo[0]:+.4f},{hi[0]:+.4f}]')
    print(f'{"":>{len(a)+len(b)+3}}  paired mF1 {d[:,1].mean():+.4f}±{d[:,1].std(ddof=1):.4f} per-seed {np.round(d[:,1],4)} | boot95 [{lo[1]:+.4f},{hi[1]:+.4f}]')
# mean and median |IG|
_, _, cols, names = load_split(log_transform=True)
ig = np.concatenate([np.load(f'{R}/xai_main_k10_ig256/xai_seed{s}.npz')['ig'] for s in range(5)])
a = np.abs(ig); mean, med = a.mean(0), np.median(a, 0)
order = np.argsort(-mean)[:8]
for j in order:
    print(f'{cols[j]:16s} mean {mean[j]:.3f} median {med[j]:.3f} p90 {np.percentile(a[:,j],90):.3f} p99 {np.percentile(a[:,j],99):.3f}')
print('top by median:', [cols[j] for j in np.argsort(-med)[:8]])
