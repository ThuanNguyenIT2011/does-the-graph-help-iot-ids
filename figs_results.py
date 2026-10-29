"""Classification figures for XGAT (5 seeds), without re-running the model.

Figure 4: row-normalised confusion matrix (recall %, mean over 5 seeds) with the number of test flows per class.
Figure 5: per-class F1 (mean ± std over 5 seeds).
Figure 6: training curves read from the log (every 20 epochs): train loss and val MCC, one line per seed.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

from figs_xai import BLUE, INK, MUTED, GRID, save   # shared style and output directory

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, 'runs', 'main_k10_5seeds')
LOG = os.path.join(ROOT, 'runs_main_k10_5seeds.log')
SEEDS = [0, 1, 2, 3, 4]
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # fixed palette order


def main():
    names = json.load(open(os.path.join(RUN, 'results.json')))['label_names']
    C = len(names)
    P = [np.load(os.path.join(RUN, f'preds_seed{s}.npz')) for s in SEEDS]
    y = P[0]['y_true']
    counts = np.bincount(y, minlength=C)

    # Figure 4
    cms = [confusion_matrix(p['y_true'], p['y_pred'], labels=range(C), normalize='true') * 100 for p in P]
    cm = np.mean(cms, 0)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#9ec5f4', '#3987e5', '#184f95'])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=100)
    for i in range(C):
        for j in range(C):
            v = cm[i, j]
            if v > 0:   # a non-zero cell that rounds to 0.0 is shown as <0.1; an empty cell means no flow
                ax.text(j, i, f'{v:.1f}' if v >= 0.05 else '<0.1', ha='center', va='center',
                        fontsize=6.5 if v >= 0.05 else 5.5, color='white' if v > 60 else INK)
    ax.set_xticks(range(C), names, rotation=40, ha='right')
    ax.set_yticks(range(C), [f'{n} ({counts[i]})' for i, n in enumerate(names)])
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class (test samples)')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, 'cm_xgat_5seeds')

    # Figure 5
    f1 = np.array([f1_score(p['y_true'], p['y_pred'], labels=range(C), average=None, zero_division=0) for p in P])
    m, sd = f1.mean(0), f1.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.bar(range(C), m, width=0.6, color=BLUE, yerr=sd, error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=1.5))
    for i, v in enumerate(m):
        ax.text(i, v + sd[i] + 0.02, f'{v:.2f}', ha='center', fontsize=6.5, color=INK)
    ax.set_xticks(range(C), names, rotation=30, ha='right')
    ax.set_ylabel('F1-score'); ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, 'f1_per_class_xgat')

    # Figure 6
    pat = re.compile(r'seed (\d+) ep (\d+) loss ([\d.]+) val_mcc ([\d.]+)')
    rec = {}
    for line in open(LOG):
        mt = pat.search(line)
        if mt:
            s, e, l, v = int(mt[1]), int(mt[2]), float(mt[3]), float(mt[4])
            rec.setdefault(s, []).append((e, l, v))
    best = {r['seed']: r['best_epoch'] for r in json.load(open(os.path.join(RUN, 'results.json')))['runs']}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for s in SEEDS:
        a = np.array(rec[s])
        axes[0].plot(a[:, 0], a[:, 1], color=CAT[s], lw=1.5, marker='o', ms=3, label=f'seed {s}')
        axes[1].plot(a[:, 0], a[:, 2], color=CAT[s], lw=1.5, marker='o', ms=3, label=f'seed {s}')
    axes[0].set_ylabel('Training loss'); axes[1].set_ylabel('Validation MCC')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=6.5, ncol=1, loc='lower right')
    fig.tight_layout()
    save(fig, 'training_curves_xgat')
    print('ok', {'f1_mean': dict(zip(names, m.round(3))), 'best_epoch': best,
                 'log_points': {s: len(rec[s]) for s in rec}})


if __name__ == '__main__':
    main()
