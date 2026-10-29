"""XAI cho XGAT: Integrated Gradients (đặc trưng) + attention GATv2 (cạnh), kèm đánh giá giải thích.

Đánh giá:
  - Fidelity+ đặc trưng: thay top-k đặc trưng (theo |IG|) bằng giá trị trung bình train (= 0 sau
    StandardScaler) rồi đo mức giảm xác suất lớp dự đoán và tỉ lệ đổi nhãn; so với che k đặc trưng ngẫu nhiên.
  - Fidelity+ cạnh: bỏ top-k láng giềng (theo attention) khỏi node đích; so với bỏ k láng giềng ngẫu nhiên.
  - Sparsity: phần khối lượng |IG| nằm trong top-5 đặc trưng.
  - Ổn định giữa seed: Spearman của độ quan trọng toàn cục; Jaccard top-5 theo từng node.
Mẫu giải thích: node test, tối đa N_PER_CLASS mỗi lớp, chọn ngẫu nhiên cố định (seed 123).
"""
import json, os, time
import numpy as np
import torch
from scipy.stats import spearmanr

from xgat import XGAT, build, pick_device

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, 'runs', 'main_k10_5seeds')
OUT = os.path.join(ROOT, 'runs', 'xai_main_k10_ig256')   # bản 32 bước (tổng Riemann phải) ở runs/xai_main_k10
SEEDS = [0, 1, 2, 3, 4]
N_PER_CLASS = 300
IG_STEPS = 256
KS = [1, 3, 5]
N_RAND = 5
BATCH = 256


def load_model(args, d_in, n_cls, seed, device):
    m = XGAT(d_in, n_cls, args['hidden'], args['layers'], args['heads'], args['dropout']).to(device)
    m.load_state_dict(torch.load(os.path.join(RUN, f'xgat_seed{seed}.pt'), map_location=device, weights_only=True))
    return m.eval()


def probs_T(model, x, nbr, ew, T, mask=None):
    with torch.no_grad():
        return torch.softmax(model.forward_targets(x, nbr, ew, T, mask), 1)


def integrated_gradients(model, x, nbr, ew, T, target_cls):
    """IG trên đặc trưng của chính node đích, baseline = 0 (trung bình train sau chuẩn hóa)."""
    x0 = x[T].clone()
    total = torch.zeros_like(x0)
    for a in (torch.arange(IG_STEPS, device=x.device) + 0.5) / IG_STEPS:   # quy tắc trung điểm
        xs = x.clone()
        inp = (a * x0).requires_grad_(True)
        xs[T] = inp
        logp = torch.log_softmax(model.forward_targets(xs, nbr, ew, T), 1)
        sel = logp.gather(1, target_cls.view(-1, 1)).sum()
        g, = torch.autograd.grad(sel, inp)
        total += g
    return (x0 * total / IG_STEPS).detach()


def main():
    device = pick_device()
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(RUN, 'results.json')))
    args = meta['args']

    class A: pass
    a = A(); a.log = args['log']; a.k = args['k']
    G = build(a, device)
    x, y = G['x'].to(device), G['y'].to(device)
    nbr, ew = G['nbr'].to(device), G['ew'].to(device)
    feat, names = G['feat_cols'], G['label_names']
    C, F_ = len(names), x.shape[1]
    test = G['masks']['test'].numpy()
    yt = G['y'].numpy()[test]
    rng = np.random.default_rng(123)
    pick = []
    for c in range(C):
        idx = test[yt == c]
        pick.append(rng.choice(idx, size=min(N_PER_CLASS, len(idx)), replace=False))
    T_all = torch.tensor(np.sort(np.concatenate(pick)), device=device)
    print(f'Giải thích {len(T_all)} node test', {names[c]: int((yt == c).sum() if False else len(p)) for c, p in enumerate(pick)}, flush=True)

    per_seed = {}
    for seed in SEEDS:
        t0 = time.time()
        model = load_model(args, F_, C, seed, device)
        # kiểm tra forward trên đồ thị con khớp forward toàn đồ thị
        with torch.no_grad():
            full = model(x, nbr, ew)[T_all[:64]]
            sub = model.forward_targets(x, nbr, ew, T_all[:64])
        assert torch.allclose(full, sub, atol=1e-3), (full - sub).abs().max()

        IG, ATT, P0, PRED = [], [], [], []
        fid_f = {k: {'top': [], 'rand': [], 'top_flip': [], 'rand_flip': []} for k in KS}
        fid_e = {k: {'top': [], 'rand': [], 'top_flip': [], 'rand_flip': []} for k in KS}
        g = torch.Generator(device='cpu').manual_seed(seed)
        for s in range(0, len(T_all), BATCH):
            T = T_all[s:s + BATCH]
            with torch.no_grad():
                logits, (a1, a2) = model.forward_targets(x, nbr, ew, T, return_att=True)
            p = torch.softmax(logits, 1); pred = p.argmax(1)
            p0 = p.gather(1, pred.view(-1, 1)).squeeze(1)
            ig = integrated_gradients(model, x, nbr, ew, T, pred)
            att = a2[:, 1:, :].mean(-1)                      # attention tầng 2 lên k láng giềng, trung bình head
            IG.append(ig.cpu()); ATT.append(att.cpu()); P0.append(p0.cpu()); PRED.append(pred.cpu())
            order_f = ig.abs().argsort(1, descending=True)
            order_e = att.argsort(1, descending=True)
            for k in KS:
                # đặc trưng: top-k
                xs = x.clone(); rows = x[T].clone()
                rows.scatter_(1, order_f[:, :k], 0.0); xs[T] = rows
                pk = probs_T(model, xs, nbr, ew, T)
                fid_f[k]['top'].append((p0 - pk.gather(1, pred.view(-1, 1)).squeeze(1)).cpu())
                fid_f[k]['top_flip'].append((pk.argmax(1) != pred).float().cpu())
                # đặc trưng: ngẫu nhiên
                for _ in range(N_RAND):
                    r = torch.rand(len(T), F_, generator=g).argsort(1)[:, :k].to(device)
                    xs = x.clone(); rows = x[T].clone(); rows.scatter_(1, r, 0.0); xs[T] = rows
                    pk = probs_T(model, xs, nbr, ew, T)
                    fid_f[k]['rand'].append((p0 - pk.gather(1, pred.view(-1, 1)).squeeze(1)).cpu())
                    fid_f[k]['rand_flip'].append((pk.argmax(1) != pred).float().cpu())
                # cạnh: bỏ top-k láng giềng
                mask = torch.ones(len(T), nbr.shape[1], dtype=torch.bool, device=device)
                mask.scatter_(1, order_e[:, :k], False)
                pk = probs_T(model, x, nbr, ew, T, mask)
                fid_e[k]['top'].append((p0 - pk.gather(1, pred.view(-1, 1)).squeeze(1)).cpu())
                fid_e[k]['top_flip'].append((pk.argmax(1) != pred).float().cpu())
                for _ in range(N_RAND):
                    r = torch.rand(len(T), nbr.shape[1], generator=g).argsort(1)[:, :k].to(device)
                    mask = torch.ones(len(T), nbr.shape[1], dtype=torch.bool, device=device); mask.scatter_(1, r, False)
                    pk = probs_T(model, x, nbr, ew, T, mask)
                    fid_e[k]['rand'].append((p0 - pk.gather(1, pred.view(-1, 1)).squeeze(1)).cpu())
                    fid_e[k]['rand_flip'].append((pk.argmax(1) != pred).float().cpu())
        IG = torch.cat(IG).numpy(); ATT = torch.cat(ATT).numpy(); PRED = torch.cat(PRED).numpy()
        absig = np.abs(IG)
        top5_mass = np.sort(absig, 1)[:, ::-1][:, :5].sum(1) / (absig.sum(1) + 1e-12)
        res = {
            'feature_fidelity': {k: {m: float(torch.cat(v).mean()) for m, v in d.items()} for k, d in fid_f.items()},
            'edge_fidelity': {k: {m: float(torch.cat(v).mean()) for m, v in d.items()} for k, d in fid_e.items()},
            'sparsity_top5_mass': float(top5_mass.mean()),
            'att_top1_mean': float(np.sort(ATT, 1)[:, -1].mean()),
            'global_importance': dict(zip(feat, absig.mean(0).tolist())),
            'per_class_importance': {names[c]: dict(zip(feat, absig[PRED == c].mean(0).tolist()))
                                     for c in range(C) if (PRED == c).any()},
            'time_s': time.time() - t0,
        }
        per_seed[seed] = res
        np.savez(os.path.join(OUT, f'xai_seed{seed}.npz'), T=T_all.cpu().numpy(), ig=IG, att=ATT, pred=PRED,
                 nbr=G['nbr'][T_all.cpu()].numpy())
        print(f'seed {seed}: {time.time() - t0:.0f}s | feat Fid+@3 top {res["feature_fidelity"][3]["top"]:.4f} '
              f'rand {res["feature_fidelity"][3]["rand"]:.4f} | edge Fid+@3 top {res["edge_fidelity"][3]["top"]:.4f} '
              f'rand {res["edge_fidelity"][3]["rand"]:.4f} | sparsity {res["sparsity_top5_mass"]:.3f}', flush=True)

    # ổn định giữa seed
    gi = np.array([[per_seed[s]['global_importance'][f] for f in feat] for s in SEEDS])
    sp = [spearmanr(gi[i], gi[j]).correlation for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
    igs = [np.load(os.path.join(OUT, f'xai_seed{s}.npz'))['ig'] for s in SEEDS]
    jac = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            ti = np.argsort(-np.abs(igs[i]), 1)[:, :5]; tj = np.argsort(-np.abs(igs[j]), 1)[:, :5]
            jac.append(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ti, tj)]))
    summary = {'n_explained': int(len(T_all)), 'features': feat, 'label_names': names,
               'stability_spearman_global': [float(np.mean(sp)), float(np.std(sp))],
               'stability_jaccard_top5_node': [float(np.mean(jac)), float(np.std(jac))],
               'per_seed': per_seed}
    json.dump(summary, open(os.path.join(OUT, 'xai_summary.json'), 'w'), indent=1)
    print('stability spearman', np.mean(sp), 'jaccard top5', np.mean(jac), flush=True)


if __name__ == '__main__':
    main()
