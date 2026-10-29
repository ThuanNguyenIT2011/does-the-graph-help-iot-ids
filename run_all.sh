#!/bin/bash
# Reproduces every experiment in the paper. Each line writes to runs/<tag>/.
set -e
cd "$(dirname "$0")"

# Detection: XGAT and the no-edge controls (five seeds, row-order split)
python xgat.py --seeds 0,1,2,3,4 --tag main_k10_5seeds
python xgat.py --ablate selfloop --seeds 0,1,2,3,4 --tag ablate_selfloop_5seeds
python xgat.py --ablate mlp --seeds 0,1,2,3,4 --tag ablate_mlp_5seeds
python knn_vote.py
python tabular_baselines.py rf
python tabular_baselines.py xgb

# Sensitivity: random class-wise split, and removal of IAT
SPLIT=random python xgat.py --seeds 0,1,2,3,4 --tag rs_xgat_5seeds
SPLIT=random python xgat.py --ablate selfloop --seeds 0,1,2,3,4 --tag rs_selfloop_5seeds
SPLIT=random python xgat.py --ablate mlp --seeds 0,1,2,3,4 --tag rs_mlp_5seeds
SPLIT=random TAG=_random python knn_vote.py
SPLIT=random TAG=_random python tabular_baselines.py rf
SPLIT=random TAG=_random python tabular_baselines.py xgb
DROP=IAT python xgat.py --ablate selfloop --seeds 0,1,2,3,4 --tag noiat_selfloop_5seeds
DROP=IAT TAG=_noiat python tabular_baselines.py xgb

# Explanations (Integrated Gradients) and their checks
python xai.py
python ig_completeness.py
python faithfulness.py
python xai_margin.py
python xai_benign.py
python xai_selfloop_rank.py
python analysis_paired.py
python figs_xai.py

# Label use in the ADA-GNetIDS anchor sampling
python anchor_leak_analysis.py
