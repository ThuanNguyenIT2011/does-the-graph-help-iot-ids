"""Shared data preparation, copying the ADA-GNetIDS protocol (main_code.py).

Kept as in the baseline: file part-00000, DoS -> DDoS, constant columns dropped, fillna(0), clip at ±1e15,
LabelEncoder, class-wise 70/10/20 split in the original row order, StandardScaler fitted on train.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
CICIOT_PATH = os.path.join(ROOT, 'data', 'Dataset2-CICIoT2023_part-00000-rouped_labels.csv')


# SPLIT=random permutes each class before cutting it (same part sizes, fixed seed 0,
# independent of the model seed); DROP=IAT,... removes these columns. The default keeps the baseline protocol.
SPLIT = os.environ.get('SPLIT', 'ordered')
DROP = [c for c in os.environ.get('DROP', '').split(',') if c]


def temporal_stratified_split(df, label_col='label_enc', train_frac=0.70, val_frac=0.10):
    # Copied verbatim from the baseline (only the SPLIT=random branch is added)
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(0)
    for _, group in df.groupby(label_col, sort=False):
        idx = group.index.to_numpy()
        if SPLIT == 'random':
            idx = rng.permutation(idx)
        n = len(idx)
        if n < 3:
            train_idx.extend(idx)
            continue
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        if n_train + n_val >= n:
            n_train, n_val = max(1, n - 2), 1
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    train_idx = np.sort(np.asarray(train_idx))
    val_idx = np.sort(np.asarray(val_idx))
    test_idx = np.sort(np.asarray(test_idx))
    return (df.loc[train_idx].reset_index(drop=True),
            df.loc[val_idx].reset_index(drop=True),
            df.loc[test_idx].reset_index(drop=True))


def signed_log1p(x):
    # Transform without fitting (no leakage): compresses the long tails of Rate, IAT, Tot sum, ...
    return np.sign(x) * np.log1p(np.abs(x))


def load_split(log_transform=False, path=CICIOT_PATH):
    df = pd.read_csv(path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df['label'] = df['label'].replace('DoS', 'DDoS')

    n_unique = df.nunique()
    const_cols = n_unique[n_unique <= 1].index.tolist()
    df = df.drop(columns=const_cols)
    df = df.fillna(0)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].clip(lower=-1e15, upper=1e15)

    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])
    label_names = list(le.classes_)
    feature_cols = [c for c in df.columns if c not in ('label', 'label_enc') and c not in DROP]

    tr, va, te = temporal_stratified_split(df)
    X = {}
    for name, part in (('train', tr), ('val', va), ('test', te)):
        a = part[feature_cols].values.astype(np.float64)
        if log_transform:
            a = signed_log1p(a)
        X[name] = a
    scaler = StandardScaler().fit(X['train'])
    out = {k: scaler.transform(v).astype(np.float32) for k, v in X.items()}
    y = {'train': tr['label_enc'].values, 'val': va['label_enc'].values, 'test': te['label_enc'].values}
    return out, y, feature_cols, label_names


def metrics(y_true, y_pred, y_prob=None, num_classes=None):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 matthews_corrcoef, roc_auc_score, confusion_matrix)
    num_classes = num_classes or int(max(y_true.max(), y_pred.max()) + 1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    far = []
    for i in range(num_classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        far.append(fp / (fp + tn) if fp + tn > 0 else 0.0)
    m = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision_w': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall_w': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_w': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'FAR': float(np.mean(far)),
    }
    if y_prob is not None:
        try:
            m['ROC_AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            m['ROC_AUC'] = float('nan')
    return m
