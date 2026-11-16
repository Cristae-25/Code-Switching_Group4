#!/usr/bin/env python
"""Re-score a gap run's saved test predictions against labels rebuilt from another sub-word parquet.

    ./.venv/bin/python scripts/analysis/rescore_preds.py xlmr_prev_gap data/subword.parquet

The model's outputs don't depend on the labels, and a relabel that only swaps
L1 and L2 keeps every scored position, so this shows what a label change does
to the score without retraining. Built for ``--decide-at gap`` runs.
Passing the parquet the run was scored on reproduces its labels (0 changed).
"""
import json, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, '.')
from src.anticipation import subword_example
from src.subword import blank_piece_ids
from src.evaluation import PAIR_KEYS
tag, sub = sys.argv[1], sys.argv[2]
d = f'data/preds/{tag}/'; g = lambda n: np.load(d + f'test_{n}.npy')
row, pos, y_old, pair, logit = g('row'), g('pos'), g('y'), g('pair'), g('logit')
tau = json.load(open(f'data/results/anticipation_{tag}.json'))['tau']
pred = (1 / (1 + np.exp(-logit.astype(np.float64))) > tau).astype(int)
blank = blank_piece_ids('xlm-roberta-base')
need = set(np.unique(row).tolist())
df = pq.read_table(sub).to_pandas()
df = df[df.row_id.isin(need)]
lab = {}
for r in df.to_dict('records'):
    ex = subword_example(r, decide_at='gap', blank_ids=blank)
    if ex is not None: lab[int(r['row_id'])] = ex['y_sw']
y_new = np.array([lab[int(r)][int(p)] for r, p in zip(row, pos)])
assert (y_new >= 0).all()
def f1(y, m):
    tp = ((pred == 1) & (y == 1) & m).sum(); return 2 * tp / max(((pred == 1) & m).sum() + ((y == 1) & m).sum(), 1)
print(f'positions {len(y_old):,}; labels changed {(y_new != y_old).sum():,}; switches {y_old.sum():,} -> {y_new.sum():,}')
print(f"{'pair':22s} {'F1 old labels':>14s} {'F1 new labels':>14s} {'changed':>8s}")
for i, k in enumerate(PAIR_KEYS):
    m = pair == i
    if m.sum(): print(f'{str(k):22s} {f1(y_old, m):>14.3f} {f1(y_new, m):>14.3f} {(y_new[m] != y_old[m]).sum():>8,}')
allm = np.ones_like(pair, bool)
print(f"{'ALL':22s} {f1(y_old, allm):>14.3f} {f1(y_new, allm):>14.3f}")
