"""Effect of label-stratified anchor sampling on the ADA-GNetIDS graph, without training.
Copies build_global_anchor_graph (main_code.py), No_ADASYN configuration: 25 mutual-information
features, scaled on the training set. Draws DRAWS anchor sets (seeds 42, 43, ...) with y_hint = y_all
and with y_hint = None. Seed 42 with labels gives the edge counts of the released code."""
import json, os, time, numpy as np
from sklearn.preprocessing import normalize
from common import load_split
N_ANCHORS, K_INTRA, K_BRIDGE, THRESH, CHUNK, DRAWS = 25_000, 15, 5, 0.5, 10_000, 20
HERE = os.path.dirname(os.path.abspath(__file__))
# Top 25 features by mutual information, as selected by the released code on this split
feats = ('IAT, Tot_size, Max, Magnitue, AVG, Tot_sum, Min, Protocol_Type, Header_Length, Weight, Variance, Std, Radius, '
         'Number, Covariance, rst_count, urg_count, Duration, flow_duration, syn_count, Srate, Rate, TCP, ack_flag_number, '
         'HTTPS').split(', ')
X, y, cols, names = load_split(log_transform=False)
sel = [cols.index(c) for c in feats]
X_all = normalize(np.vstack([X['train'], X['val'], X['test']])[:, sel].astype(np.float32))
y_all = np.concatenate([y['train'], y['val'], y['test']]); N = len(y_all)
n_tr, n_va = len(y['train']), len(y['val'])
held = np.arange(N) >= n_tr; test = np.arange(N) >= n_tr + n_va

def anchors(y_hint, seed):
    if y_hint is not None:
        rng = np.random.default_rng(seed); idx = []
        classes, counts = np.unique(y_hint, return_counts=True)
        for cls, cnt in zip(classes, counts):
            ci = np.where(y_hint == cls)[0]
            take = max(1, int(round(N_ANCHORS * cnt / N)))
            idx.append(rng.choice(ci, size=min(take, len(ci)), replace=False))
        idx = np.concatenate(idx)
        if len(idx) > N_ANCHORS: idx = idx[:N_ANCHORS]
        elif len(idx) < N_ANCHORS:
            idx = np.concatenate([idx, np.random.default_rng(seed + 1).choice(N, size=N_ANCHORS - len(idx), replace=False)])
    else:
        idx = np.random.default_rng(seed).choice(N, size=N_ANCHORS, replace=False)
    return np.unique(idx)

def topk(Q, B, k):
    S = Q @ B.T
    i = np.argpartition(-S, k - 1, axis=1)[:, :k]
    v = np.take_along_axis(S, i, 1); o = np.argsort(-v, axis=1)
    return np.take_along_axis(v, o, 1), np.take_along_axis(i, o, 1)

def graph(a):
    B = X_all[a]; src, dst = [], []
    for s in range(0, len(a), CHUNK):
        v, i = topk(B[s:s + CHUNK], B, K_INTRA + 1)
        # drop the first column (the node itself), as idxs_a[:, 1:] in the released code
        r, c = np.where(v[:, 1:] >= THRESH)
        src.append(a[s + r]); dst.append(a[i[:, 1:][r, c]])
    n_back = sum(map(len, src))
    na = np.setdiff1d(np.arange(N), a)
    for s in range(0, len(na), CHUNK):
        ch = na[s:s + CHUNK]; v, i = topk(X_all[ch], B, K_BRIDGE)
        r, c = np.where(v >= THRESH)
        src.append(ch[r]); dst.append(a[i[r, c]])
    u, w = np.concatenate(src), np.concatenate(dst)
    n_bridge = len(u) - n_back
    return np.concatenate([u, w]), np.concatenate([w, u]), n_back, n_bridge  # both directions, as in the released code

def stats(a, s, d, n_back, n_bridge):
    r = {'backbone': int(n_back), 'bridge': int(n_bridge), 'edges': len(s)}
    r['anchors'] = np.bincount(y_all[a], minlength=len(names)).tolist()
    r['heldout_anchor_share'] = float(held[a].mean())
    r['edge_touch_heldout'] = float((held[s] | held[d]).mean())
    r['edge_both_heldout'] = float((held[s] & held[d]).mean())
    into = test[d]; same = y_all[s[into]] == y_all[d[into]]
    r['test_same_class'] = float(same.mean())
    per = [float(same[y_all[d[into]] == c].mean()) for c in range(len(names))]
    r['test_same_class_per'] = per; r['test_same_class_macro'] = float(np.mean(per))
    r['test_no_edge'] = float(1 - np.isin(np.where(test)[0], np.unique(d[into])).mean())
    return r

out = {'names': names, 'with_labels': [], 'no_labels': []}
for k, hint in (('with_labels', y_all), ('no_labels', None)):
    for i in range(DRAWS):
        t = time.time(); a = anchors(hint, 42 + i); s, d, nb, nbr = graph(a)
        r = stats(a, s, d, nb, nbr); r['seed'] = 42 + i; out[k].append(r)
        print(k, r['seed'], nb, nbr, round(r['test_same_class'], 5), round(r['test_same_class_macro'], 4), f'{time.time() - t:.1f}s', flush=True)
        os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
        json.dump(out, open(os.path.join(HERE, 'runs/anchor_graph_audit.json'), 'w'), indent=1)
