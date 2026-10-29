"""Vẽ hình minh họa XAI (Integrated Gradients) cho bài ICCIES. Đọc kết quả của xai.py, không chạy lại model.

Hình 1: độ quan trọng toàn cục (mean |IG|), top-15, thanh lỗi = độ lệch chuẩn qua 5 seed.
Hình 2: heatmap lớp thật × 12 đặc trưng, chỉ flow dự đoán đúng (n = số flow trung bình mỗi seed).
Hình 3: giải thích cục bộ cho một flow dự đoán đúng của mỗi lớp (seed 0), IG có dấu, top-8.
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

    # Hình 1
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

    # Hình 2
    top = np.argsort(-m)[:12]
    # chỉ các flow mà model của seed đó dự đoán đúng; IG giải thích lớp dự đoán nên khi đúng nó cũng là lớp thật
    preds = np.stack([d['pred'] for d in D])                      # (seed, node)
    correct = preds == yt[None, :]
    H, n_avg = [], []
    for c in range(len(names)):
        sel = correct & (yt[None, :] == c)                        # (seed, node)
        H.append(absig[:, :, top][sel].mean(0))
        n_avg.append(sel.sum() / len(SEEDS))
    H = np.stack(H)
    # bỏ hàng DDoS: baseline 0 đã được dự đoán là DDoS với p ≈ 1 nên tổng IG ≈ 1e-6, chuẩn hóa theo hàng chỉ phóng đại nhiễu
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

    # Hình 3: seed 0, mỗi lớp một flow dự đoán đúng có độ quan trọng tổng ở trung vị (không chọn ví dụ đẹp nhất)
    ig0, pred0 = D[0]['ig'], D[0]['pred']
    # chỉ 4 lớp có đủ flow dự đoán đúng; bỏ DDoS (log-xác suất bão hòa), Brute Force (1 flow), Web-Based (vài flow)
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
