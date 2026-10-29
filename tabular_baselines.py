"""Random Forest và XGBoost trên cùng split và tiền xử lý (signed log + chuẩn hóa train),
5 seed. Chọn số cây/số vòng theo val MCC không áp dụng: dùng cấu hình mặc định cố định, không tune trên test.
Trọng số lớp giống XGAT (1/sqrt(n_c), chuẩn hóa) để so cùng điều kiện."""
import json, os, sys, time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import load_split, metrics
X, y, cols, names = load_split(log_transform=True)
C = len(names)
cnt = np.bincount(y['train'], minlength=C).astype(float)
cw = 1 / np.sqrt(cnt); cw = cw / cw.sum() * C
w_tr = cw[y['train']]
which = sys.argv[1] if len(sys.argv) > 1 else 'rf'
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', f'tab_{which}_5seeds' + os.environ.get('TAG', '')); os.makedirs(out, exist_ok=True)
rows = []
for s in range(5):
    t = time.time()
    if which == 'rf':
        m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s)
        m.fit(X['train'], y['train'], sample_weight=w_tr)
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
                              colsample_bytree=0.9, tree_method='hist', n_jobs=-1, random_state=s,
                              early_stopping_rounds=30, eval_metric='mlogloss')
        m.fit(X['train'], y['train'], sample_weight=w_tr, eval_set=[(X['val'], y['val'])], verbose=False)
    prob = m.predict_proba(X['test']); yp = prob.argmax(1); yt = y['test']
    r = metrics(yt, yp, prob, C); r.update(seed=s, train_s=time.time() - t,
                                       benign_FA=float(((yt == 0) & (yp != 0)).sum() / (yt == 0).sum()))
    np.savez(os.path.join(out, f'preds_seed{s}.npz'), y_true=yt, y_pred=yp, y_prob=prob)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    rows.append(r)
json.dump({'model': which, 'label_names': names, 'runs': rows}, open(os.path.join(out, 'results.json'), 'w'), indent=1)
