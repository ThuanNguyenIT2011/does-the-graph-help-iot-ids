"""kNN majority vote trên đúng đồ thị của XGAT (k=10 láng giềng train, cosine).
Không có tham số học và không ngẫu nhiên; hòa phiếu chọn lớp có tổng cosine lớn hơn."""
import numpy as np, torch, json, os
from xgat import build, pick_device
from common import metrics
class A: log = 1; k = 10
G = build(A(), pick_device())
y = G['y'].numpy(); nbr = G['nbr'].numpy(); sim = G['ew'].squeeze(-1).numpy(); te = G['masks']['test'].numpy()
C = len(G['label_names'])
votes = np.zeros((len(te), C)); 
for j in range(nbr.shape[1]):
    np.add.at(votes, (np.arange(len(te)), y[nbr[te, j]]), 1 + 1e-6 * sim[te, j])
prob = votes / votes.sum(1, keepdims=True)
r = metrics(y[te], prob.argmax(1), prob, C)
yt, yp = y[te], prob.argmax(1)
r['benign_FA'] = float(((yt == 0) & (yp != 0)).sum() / (yt == 0).sum())
print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
json.dump(r, open(f"runs/knn_vote_k10{os.environ.get('TAG', '')}.json", 'w'), indent=1)
