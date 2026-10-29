"""cách ADA-GNetIDS chọn anchor khi y_hint = y_all so với khi không dùng nhãn.
Chép đúng nhánh chọn anchor của build_global_anchor_graph (main_code.py dòng 229-249), cấu hình
No_ADASYN (không có mẫu tổng hợp nên thứ tự node là train, val, test như code gốc)."""
import numpy as np
from common import load_split
N_ANCHORS = 25_000
_, y, _, names = load_split(log_transform=False)
y_all = np.concatenate([y['train'], y['val'], y['test']]); N = len(y_all)
n_tr, n_va = len(y['train']), len(y['val'])
part = np.array(['train'] * n_tr + ['val'] * n_va + ['test'] * (N - n_tr - n_va))

def anchors(y_hint):
    n_anchors = min(N_ANCHORS, N)
    if y_hint is not None:
        rng = np.random.default_rng(42); idx = []
        classes, counts = np.unique(y_hint, return_counts=True)
        for cls, cnt in zip(classes, counts):
            ci = np.where(y_hint == cls)[0]
            take = max(1, int(round(n_anchors * cnt / N)))
            idx.append(rng.choice(ci, size=min(take, len(ci)), replace=False))
        idx = np.concatenate(idx)
        if len(idx) > n_anchors: idx = idx[:n_anchors]
        elif len(idx) < n_anchors:
            idx = np.concatenate([idx, np.random.default_rng(43).choice(N, size=n_anchors - len(idx), replace=False)])
    else:
        idx = np.random.default_rng(42).choice(N, size=n_anchors, replace=False)
    return np.unique(idx)

A = {'y_all': anchors(y_all), 'none': anchors(None)}
print(f'N={N}, anchors: ' + ', '.join(f'{k} {len(v)}' for k, v in A.items()))
print(f'{"class":12s} {"N":>7s} | y_all: anchors (val+test) | none: anchors (val+test)')
for c, n in enumerate(names):
    row = []
    for k in A:
        a = A[k]; m = y_all[a] == c
        row.append(f'{m.sum():6d} ({np.isin(part[a][m], ["val","test"]).sum():4d})')
    print(f'{n:12s} {(y_all==c).sum():7d} | {row[0]:>24s} | {row[1]:>22s}')
for k, a in A.items():
    print(k, 'held-out share of anchors %.3f' % np.isin(part[a], ['val', 'test']).mean())
