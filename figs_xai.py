"""XAI figures (Integrated Gradients) for the ICCIES paper. Reads the output of xai.py, does not re-run the model.

Figure 1: global importance (mean |IG|), top 15, error bars = standard deviation over 5 seeds.
Figure 2: heatmap of true class x 12 features, correctly predicted flows only (n = mean number of flows per seed).
Figure 3: local explanation of one correctly predicted flow per class (seed 0), signed IG, top 8.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load_split

ROOT = os.path.dirname(os.path.abspath(__file__))
XAI = os.path.join(ROOT, 'runs', 'xai_main_k10_ig256')
OUT = os.path.join(ROOT, '..', 'paper', 'iccies2027', 'figs')
SEEDS = [0, 1, 2, 3, 4]

BLUE, RED, INK, MUTED, GRID = '#256abf', '#e34948', '#222222', '#6b6b6b', '#e6e6e3'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)


def main():
    S = json.load(open(os.path.join(XAI, 'xai_summary.json')))
    feat, names = S['features'], S['label_names']
    _, y, _, _ = load_split(log_transform=True)
    y_all = np.concatenate([y['train'], y['val'], y['test']])
    D = [np.load(os.path.join(XAI, f'xai_seed{s}.npz')) for s in SEEDS]
    T = D[0]['T']; yt = y_all[T]
    absig = np.stack([np.abs(d['ig']) for d in D])            # (seed, node, feat)

    # Figure 1
    per_seed = absig.mean(1)                                    # (seed, feat)
    m, sd = per_seed.mean(0), per_seed.std(0, ddof=1)
    o = np.argsort(-m)[:15][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh(range(len(o)), m[o], height=0.6, color=BLUE, xerr=sd[o],
            error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=1.5))
    ax.set_yticks(range(len(o)), [feat[i].replace('_', ' ') for i in o])
    ax.set_xlabel('Mean |IG| attribution')
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, 'xai_global_importance')

    # Figure 2
    top = np.argsort(-m)[:12]
    # only flows that the model of that seed predicts correctly; IG explains the predicted class, which is then the true class
    preds = np.stack([d['pred'] for d in D])                      # (seed, node)
    correct = preds == yt[None, :]
    H, n_avg = [], []
    for c in range(len(names)):
        sel = correct & (yt[None, :] == c)                        # (seed, node)
        H.append(absig[:, :, top][sel].mean(0))
        n_avg.append(sel.sum() / len(SEEDS))
    H = np.stack(H)
    # drop the DDoS row: the zero baseline is already predicted as DDoS with p ≈ 1, so sum IG ≈ 1e-6 and row normalisation only amplifies noise
    rows = [c for c in range(len(names)) if names[c] != 'DDoS']
    H = H[rows]
    Hn = H / H.max(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    im = ax.imshow(Hn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'blues', ['#f4f8fd', '#9ec5f4', '#3987e5', '#184f95']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(top)), [feat[i].replace('_', ' ') for i in top], rotation=45, ha='right')
    ax.set_yticks(range(len(rows)), [f'{names[c]} (n={n_avg[c]:.1f})' for c in rows])
    ax.set_ylabel('Class (n: mean over 5 models)', fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Relative |IG| (row max = 1)', fontsize=7); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    save(fig, 'xai_class_heatmap')

    # Figure 3: seed 0, one correctly predicted flow per class whose total importance is at the median (not the nicest example)
    ig0, pred0 = D[0]['ig'], D[0]['pred']
    # only 4 classes have enough correctly predicted flows; DDoS (saturated log-probability), Brute Force (1 flow), Web-Based (a few flows) are dropped
    show = [names.index(n) for n in ('Benign', 'Mirai', 'Recon', 'Spoofing')]
    ncol = 2; nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 4.4))
    axes = np.atleast_2d(axes)
    picked = {}
    for ax, c in zip(axes.flat, show):
        idx = np.where((yt == c) & (pred0 == c))[0]
        tot = np.abs(ig0[idx]).sum(1)
        i = idx[np.argsort(tot)[len(idx) // 2]]
        picked[names[c]] = int(T[i])
        v = ig0[i]; o = np.argsort(-np.abs(v))[:8][::-1]
        ax.barh(range(8), v[o], height=0.6, color=[BLUE if x > 0 else RED for x in v[o]])
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_yticks(range(8), [feat[j].replace('_', ' ') for j in o], fontsize=8.5)
        ax.set_title(f'{names[c]} (node {int(T[i])})', loc='left', fontsize=9.5)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=8)
    for ax in list(axes.flat)[len(show):]:
        ax.axis('off')
    fig.supxlabel('IG attribution to the predicted class (log-probability); blue supports, red opposes', fontsize=8.5)
    fig.tight_layout()
    save(fig, 'xai_local_examples')
    json.dump({'local_example_nodes': picked, 'top15_global': [feat[i] for i in np.argsort(-m)[:15]],
               'class_counts': {n: int((yt == c).sum()) for c, n in enumerate(names)}},
              open(os.path.join(XAI, 'figs_meta.json'), 'w'), indent=1)
    print('ok', picked)


if __name__ == '__main__':
    main()
