#!/usr/bin/env python
"""Switch F1 split by whether the two taggers agree on the words around each label (no GPU).

    ./.venv/bin/python scripts/consensus_corpus.py        # -> data/consensus.parquet (once)
    ./.venv/bin/python scripts/consensus_f1.py xlmr_prev_gap

Each scored position is tied to the word before it and the word after it
(via ``piece_word`` in data/subword.parquet). It is ``agree`` when production
(AnE-LID) and dict-freq agree on both words, ``disagree`` when either word is
split. Only ``ane`` rows are covered; German->French and Devanagari Hindi
have no independent check. dict-freq is weak on romanized Hindi, so a Hindi
disagreement says little. Writes data/consensus_f1_<TAG>.json.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.anticipation import NONE
from src.evaluation import PAIR_KEYS
tag = sys.argv[1]
d = f'data/preds_{tag}/'; g = lambda n: np.load(d + f'test_{n}.npy')
row, pos, y, pair, logit = g('row'), g('pos'), g('y'), g('pair'), g('logit')
tau = json.load(open(f'data/anticipation_{tag}.json'))['tau']
pred = (1 / (1 + np.exp(-logit.astype(np.float64))) > tau).astype(int)
c = pd.read_parquet('data/consensus.parquet').set_index('row_id')['consensus']
s = pd.read_parquet('data/subword.parquet', columns=['row_id', 'piece_word', 'piece_lang', 'y_sw']).set_index('row_id')
bucket = np.full(len(y), 'unchecked', dtype=object)
mism = 0; checked = 0
order = np.argsort(row, kind='stable')
for r, idx in pd.Series(order).groupby(row[order]):
    if r not in c.index: continue
    cons = c.loc[r]; pw = np.asarray(s.loc[r, 'piece_word']); pl = np.asarray(s.loc[r, 'piece_lang'])
    if tag == 'xlmr_prev':                     # sanity: pos is the piece index
        ys = np.asarray(s.loc[r, 'y_sw']); p = pos[idx.values]
        ok = p < len(ys); checked += ok.sum(); mism += (ys[p[ok]] != y[idx.values][ok]).sum()
    lb = np.flatnonzero(pl != NONE)
    for i in idx.values:
        j = pos[i]
        prev = lb[lb <= j]; nxt = lb[lb > j]
        if not len(prev) or not len(nxt): continue
        a, b = pw[prev[-1]], pw[nxt[0]]
        ca, cb = cons[a], cons[b]
        bucket[i] = 'agree' if ca == 'agree' and cb == 'agree' else ('disagree' if 'disagree' in (ca, cb) else 'abstain')
if tag == 'xlmr_prev': print(f'sanity: {mism} label mismatches of {checked:,} checked positions')
def prf(m):
    tp = ((pred == 1) & (y == 1) & m).sum(); P = tp / max(((pred == 1) & m).sum(), 1); R = tp / max(((y == 1) & m).sum(), 1)
    return P, R, 2 * P * R / max(P + R, 1e-12), m.sum(), y[m].mean() if m.sum() else 0
print(f'== {tag}')
print(f"{'pair':22s} {'bucket':10s} {'n':>9s} {'base':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
out = {}
for i, k in enumerate(PAIR_KEYS):
    pm = pair == i
    if not (pm & (bucket != 'unchecked')).any(): continue
    for bk in ('all', 'agree', 'disagree'):
        m = pm & (bucket != 'unchecked') if bk == 'all' else pm & (bucket == bk)
        P, R, F, n, base = prf(m)
        out.setdefault(str(k), {})[bk] = dict(n=int(n), base=float(base), P=float(P), R=float(R), F1=float(F))
        print(f'{str(k):22s} {bk:10s} {n:>9,} {base:>6.3f} {P:>6.3f} {R:>6.3f} {F:>6.3f}')
allm = bucket != 'unchecked'
for bk in ('all', 'agree', 'disagree'):
    m = allm if bk == 'all' else bucket == bk
    P, R, F, n, base = prf(m); out.setdefault('ALL', {})[bk] = dict(n=int(n), base=float(base), P=float(P), R=float(R), F1=float(F))
    print(f"{'ALL checked':22s} {bk:10s} {n:>9,} {base:>6.3f} {P:>6.3f} {R:>6.3f} {F:>6.3f}")
json.dump(out, open(f'data/consensus_f1_{tag}.json', 'w'), indent=1)
