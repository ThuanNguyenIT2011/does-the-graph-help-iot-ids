# Does the Graph Help? Label Leakage and a No-Edge Control for Similarity-Graph GNNs in IoT Intrusion Detection

Code for the paper by Phan Thanh Hy and Nguyen Van Thuan (Posts and Telecommunications Institute of
Technology, Ho Chi Minh City), submitted to ICCIES 2027.

The paper asks two questions about graph neural networks that build their graph from feature
similarity on CICIoT2023: whether the graph construction uses held-out labels, and whether the
graph improves detection. It compares a label-free graph attention detector (XGAT) with controls
that remove the edges (self-loop, MLP), use only the neighbour labels (kNN vote), or drop the neural
network (Random Forest, XGBoost), and explains XGAT with Integrated Gradients.

## Setup

```bash
pip install -r requirements.txt
```

Put the data file in `data/` as described in [data/README.md](data/README.md). All scripts pick
MPS, then CUDA, then CPU; set `CUDA_VISIBLE_DEVICES=-1` to force the CPU (XGAT on the full graph
needs more than 4 GB of GPU memory).

## Files

| File | Purpose |
|---|---|
| `common.py` | Loading, preprocessing (signed log, standardisation on train), class-wise 70/10/20 split, metrics |
| `xgat.py` | Label-free cosine kNN graph (k = 10, training neighbours only) and the XGAT model; `--ablate selfloop` masks all neighbour edges, `--ablate mlp` removes the attention layers |
| `knn_vote.py` | Majority vote over the same ten training neighbours |
| `tabular_baselines.py` | Random Forest (`rf`) and XGBoost (`xgb`) on the same features and split |
| `xai.py` | Integrated Gradients (zero baseline, 256 steps) and attention for XGAT |
| `ig_completeness.py`, `faithfulness.py` | Completeness, deletion, and insertion tests |
| `xai_margin.py`, `xai_benign.py` | IG on the logit margin, with zero and benign-mean baselines |
| `xai_selfloop_rank.py` | IG ranking of the self-loop model and Spearman correlation with XGAT |
| `analysis_paired.py` | Paired per-seed differences, bootstrap intervals, mean and median \|IG\| |
| `anchor_leak_analysis.py` | Reproduces the ADA-GNetIDS anchor sampling with and without labels |
| `figs_xai.py`, `figs_results.py` | Figures |
| `run_all.sh` | Every command used for the paper |
| `results/` | `results.json` of the runs reported in the paper (metrics per seed and arguments) |

Two environment variables change the protocol; the default reproduces the main results.

- `SPLIT=random` permutes each class with a fixed seed before cutting it (same part sizes).
- `DROP=IAT` removes the listed features (comma-separated).

## Results

MCC, mean ± standard deviation over five seeds (kNN vote is deterministic).

| Model | Row-order split | Random split | Row-order, no IAT |
|---|---|---|---|
| XGAT | 0.9333 ± 0.0051 | 0.9380 ± 0.0022 | – |
| Self-loop | 0.9389 ± 0.0017 | 0.9394 ± 0.0013 | 0.9409 ± 0.0024 |
| MLP | 0.9281 ± 0.0052 | 0.9377 ± 0.0023 | – |
| kNN vote | 0.9405 | 0.9431 | – |
| Random Forest | 0.9704 ± 0.0002 | 0.9703 ± 0.0011 | – |
| XGBoost | 0.9743 ± 0.0004 | 0.9724 ± 0.0007 | 0.9507 ± 0.0004 |

The no-IAT self-loop run was interrupted after two seeds; `results/noiat_selfloop/results_seeds234.json`
holds seeds 2 to 4, and the value above combines all five seeds.

Code comments are partly in Vietnamese.
