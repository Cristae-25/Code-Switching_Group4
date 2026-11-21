#!/usr/bin/env python
"""Where does a final model make its switch errors, per language? (no GPU)

    ./.venv/bin/python scripts/analysis/error_breakdown.py xlmr_prev_gap_36k_rsv2
    ./.venv/bin/python scripts/analysis/error_breakdown.py mbert_prev_gap_36k_rsv2 --path data/subword_mbert.parquet

This reads the saved test predictions (``data/preds/<TAG>``) at the run's own tau and
slices them per language pair four ways:

* ``cs_type``: SwitchLingua's own generation label (Intrasentential / Intersentential /
  Extra-sentential tag), joined from ``data/tagged.parquet`` on ``row_id``. It is the
  type the generator was *asked* for, not a verified one, so the script first
  cross-tabulates it against the observed gap buckets.
* gap bucket (``punct_breakdown``): what lies between the decision word and the next
  language-bearing piece. ``adjacent`` is a mid-clause switch, ``sentence_end`` / ``turn``
  are inter-sentential.
* direction: into L2 (switch out of L1) vs back to L1.
* L1 clause ending: the text read so far ends in a sentence-final ending, a connective,
  a particle (Korean, Japanese, Chinese, Cantonese; suffix heuristics in ``ENDINGS``). This
  feature is causal: it uses pieces <= t only, so it is something the model can see.

It also gives a **mix-standardised F1** per pair: positions are reweighted so that each
pair has the pooled share of every (bucket, label) cell. A pair that stays on top after
standardisation is better *within* each kind of switch, not just luckier in its mix.

Duration (3-class macro-F1) is sliced by ``cs_type``. Writes
``data/results/error_breakdown_<TAG>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.punct_breakdown import BUCKETS, NO_LABEL, gap_buckets, load_preds  # noqa: E402
from src.anticipation import NONE, gap_landing  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402
from src.metrics import duration_scores, fmt_table, prf, wilson  # noqa: E402

CS_TYPES = {"Intrasentential": "intra", "Intersentential": "inter",
            "Extra-sentential / Tag switching": "extra"}
CATS = ("final", "connective", "particle", "informal", "other")
NO_CAT = -1

# Suffixes of the L1 text read so far. Longest match wins. These are heuristics: a
# Korean word ending in 다 is usually a declarative verb but can be a noun (바다).
ENDINGS = {
    "Korean": {
        "final": ["니다", "니까", "어요", "아요", "에요", "예요", "세요", "해요", "네요", "군요", "죠",
                  "지요", "거든", "거든요", "다", "요", "까", "자", "래", "대", "냐", "니"],
        "connective": ["는데", "은데", "고", "서", "면", "지만", "으니", "며", "면서", "도록", "려고",
                       "는데도", "어서", "아서", "니까는", "든지", "거나"],
        "particle": ["은", "는", "이", "가", "을", "를", "에", "에서", "의", "도", "로", "으로", "와",
                     "과", "랑", "이랑", "한테", "에게", "만", "까지", "부터", "보다", "처럼"],
        "informal": ["어", "아", "해", "워", "봐", "줘", "돼", "야", "지"],
    },
    "Japanese": {
        "final": ["ます", "です", "ました", "でした", "ません", "ましょう", "だ", "ね", "よ", "か",
                  "よね", "だよ", "だね", "な", "わ", "の"],
        "connective": ["けど", "けれど", "から", "ので", "のに", "て", "って", "たら", "ば"],
        "particle": ["は", "を", "に", "が", "で", "と", "も", "へ"],
    },
    "Chinese": {
        "final": ["了", "吗", "呢", "吧", "啊", "呀", "嘛"],
        "connective": ["但是", "所以", "因为", "而且", "然后", "可是"],
    },
    "Cantonese": {
        "final": ["啦", "喇", "呀", "咩", "囉", "喎", "嘛", "啫", "㗎", "嘅", "呢", "吖", "啊", "咋"],
        "connective": ["但係", "所以", "因為", "跟住", "然後", "不過"],
    },
}


def _suffix_table(l1: str):
    table = [(s, CATS.index(c)) for c, ss in ENDINGS.get(l1, {}).items() for s in ss]
    return sorted(table, key=lambda t: -len(t[0]))


def classify(text: str, table) -> int:
    """Category index of the longest listed suffix of ``text``, else ``other``."""
    for s, c in table:
        if text.endswith(s):
            return c
    return CATS.index("other")


def strip_piece(p: str) -> str:
    return p.replace("▁", "").replace("##", "").strip()


def is_word_start(p: str, scheme: str) -> bool:
    """XLM-R marks a word start with ``▁``; WordPiece marks a continuation with ``##``."""
    return p.startswith("▁") if scheme == "sentencepiece" else not p.startswith("##")


def ending_categories(pieces, langs, l1: str, scheme: str) -> np.ndarray:
    """Per piece, the ending category of the L1 word read so far (L1 pieces only).

    Causal: the category at piece t uses pieces <= t. A non-L1 piece (L2, punctuation)
    ends the current word.
    """
    table = _suffix_table(l1)
    out = np.full(len(langs), NO_CAT, dtype=np.int8)
    if not table:
        return out
    buf = ""
    for i, (p, lang) in enumerate(zip(pieces, langs)):
        if lang != 0:
            buf = ""
            continue
        t = strip_piece(p)
        buf = t if is_word_start(p, scheme) else buf + t
        out[i] = classify(buf, table)
    return out


def relocate(mapping: dict, *arrays, fill=NO_LABEL):
    """Apply a ``gap_landing`` map to several per-piece arrays (labels, features)."""
    outs = [a.copy() for a in arrays]
    for src, dst in mapping.items():
        for a, o in zip(arrays, outs):
            o[dst] = a[src]
            o[src] = fill
    return outs


def standardised_f1(y, pred, cell, ref_share: np.ndarray) -> float:
    """F1 after reweighting positions so cell shares (bucket x label) equal ``ref_share``."""
    y, pred, cell = np.asarray(y, bool), np.asarray(pred, bool), np.asarray(cell)
    n = np.bincount(cell, minlength=len(ref_share)).astype(float)
    own = n / n.sum()
    w_cell = np.divide(ref_share, own, out=np.zeros_like(ref_share), where=own > 0)
    w = w_cell[cell]
    tp, fp, fn = w[y & pred].sum(), w[~y & pred].sum(), w[y & ~pred].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def scan_test(paths, scheme: str, blank_ids, test_only: bool = True):
    """``{row_id: (bucket, y, ending)}`` per row (test rows only by default), at the scored positions."""
    out = {}
    for path in ([paths] if isinstance(paths, str) else paths):
        _scan_one(path, scheme, blank_ids, test_only, out)
    return out


def _scan_one(path, scheme, blank_ids, test_only, out):
    f = pq.ParquetFile(path)
    cols = ["row_id", "first_language", "pieces", "piece_lang", "y_sw", "piece_utt", "piece_ids"]
    for batch in f.iter_batches(batch_size=4096, columns=cols):
        d = batch.to_pydict()
        for i, rid in enumerate(d["row_id"]):
            if test_only and rid % 10 != 9:
                continue
            langs = np.asarray(d["piece_lang"][i])
            b = gap_buckets(d["pieces"][i], langs, d["piece_utt"][i])
            y = np.asarray(d["y_sw"][i], dtype=np.int8)
            e = ending_categories(d["pieces"][i], langs, d["first_language"][i], scheme)
            if blank_ids is not None:
                m = gap_landing(langs, np.isin(np.asarray(d["piece_ids"][i]), blank_ids))
                b, y, e = relocate(m, b, y, e)
            out[rid] = (b, y, e)


def cell_stats(y, pred, total_fn: int, total_fp: int) -> dict:
    s = prf(y, pred)
    s["base_rate"] = float(np.mean(y)) if len(y) else 0.0
    s["mean_prob"] = None
    s["share_of_fn"] = s["fn"] / total_fn if total_fn else 0.0
    s["share_of_fp"] = s["fp"] / total_fp if total_fp else 0.0
    s["base_rate_ci"] = wilson(int(np.sum(y)), int(len(y)))
    return s


def slice_by(y, pred, prob, key, names) -> dict:
    tot_fn, tot_fp = int(np.sum(y & ~pred)), int(np.sum(~y & pred))
    out = {}
    for j, name in enumerate(names):
        m = key == j
        if not m.any():
            continue
        s = cell_stats(y[m], pred[m], tot_fn, tot_fp)
        s["mean_prob"] = float(prob[m].mean())
        out[name] = s
    return out


def load_switchlingua(tag: str, path: str, tagged: str, scheme: str, blank_ids):
    """Saved test predictions joined to buckets, endings and cs_type."""
    z = load_preds(tag)
    lookup = scan_test(path, scheme, blank_ids)
    meta = pd.read_parquet(tagged, columns=["row_id", "cs_type"]).set_index("row_id")["cs_type"]
    ct_names = list(CS_TYPES.values())
    ct_of_row = meta.map(lambda s: ct_names.index(CS_TYPES[s]) if s in CS_TYPES else -1)
    return {k[5:]: v for k, v in z.items() if k.startswith("test_")}, lookup, ct_of_row


def load_lince(tag: str, backbone: str, scheme: str, blank_ids):
    """Saved LinCE predictions (``eval_lince.py --save-preds``), every split, joined to buckets."""
    from src import lince
    z = dict(np.load(f"data/preds/lince_{tag}.npz"))
    splits = sorted({k.split("_", 1)[0] for k in z})
    keys = {k.split("_", 1)[1] for k in z}
    arrays = {k: np.concatenate([z[f"{s}_{k}"] for s in splits]) for k in keys}
    paths = [lince.parquet_path(backbone, p, s) for p in lince.PAIRS for s in splits
             if os.path.exists(lince.parquet_path(backbone, p, s))]
    return arrays, scan_test(paths, scheme, blank_ids, test_only=False), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--path", default="data/subword.parquet")
    ap.add_argument("--tagged", default="data/tagged.parquet")
    ap.add_argument("--lince", action="store_true",
                    help="slice data/preds/lince_<TAG>.npz (LinCE tweets) instead of the SwitchLingua test split")
    args = ap.parse_args()

    run = json.load(open(f"data/results/anticipation_{args.tag}.json"))
    ra = run["args"]
    if ra.get("turn_sep"):
        raise SystemExit("--turn-sep runs shift piece positions; not supported")
    from src.xlmr_model import BACKBONES
    backbone = ra.get("backbone") or BACKBONES[ra["model"]]
    scheme = "wordpiece" if "bert-base" in backbone and "roberta" not in backbone else "sentencepiece"
    blank_ids = None
    if ra.get("decide_at", "word") == "gap":
        from src.subword import blank_piece_ids
        blank_ids = blank_piece_ids(backbone)
    tau = float(run["tau"])                 # eval_lince also scores at the SwitchLingua-val tau
    print(f"{args.tag}: {backbone} ({scheme}) | tau {tau:.3f} | decide_at {ra.get('decide_at', 'word')}"
          f"{' | LinCE' if args.lince else ''}")

    if args.lince:
        z, lookup, ct_of_row = load_lince(args.tag, backbone, scheme, blank_ids)
    else:
        z, lookup, ct_of_row = load_switchlingua(args.tag, args.path, args.tagged, scheme, blank_ids)
    rows, pos, y = z["row"], z["pos"], z["y"].astype(bool)
    prob = 1 / (1 + np.exp(-z["logit"]))
    pred = prob >= tau
    pair = np.array(PAIR_KEYS)[z["pair"]]
    direction = z["lang"].astype(int)          # 0 = currently L1 (into L2), 1 = into L1

    bucket = np.empty(len(y), np.int8)
    ending = np.empty(len(y), np.int8)
    labels = np.empty(len(y), np.int8)
    for k, (r, p) in enumerate(zip(rows.tolist(), pos.tolist())):
        b, yy, e = lookup[r]
        bucket[k], labels[k], ending[k] = b[p], yy[p], e[p]
    if not np.array_equal(labels.astype(bool), y):
        raise SystemExit("parquet y_sw disagrees with the saved predictions (wrong --path?)")
    if (bucket == NO_LABEL).any():
        raise SystemExit(f"{int((bucket == NO_LABEL).sum())} scored positions have no bucket")
    overall = prf(y, pred)
    if args.lince:
        print(f"pooled F1 {overall['f1']:.4f} (lince json 'all': "
              f"{json.load(open(f'data/results/lince_{args.tag}.json'))['all']['overall']['f1']:.4f})")
    else:
        print(f"pooled F1 {overall['f1']:.4f} (run json: {run['overall']['f1']:.4f})")

    ct_names = list(CS_TYPES.values())
    cstype = None
    if ct_of_row is not None:
        cstype = ct_of_row.reindex(rows).fillna(-1).astype(int).to_numpy()

    # pooled (bucket, label) mix for standardisation
    cell = bucket.astype(int) * 2 + y.astype(int)
    ref = np.bincount(cell, minlength=2 * len(BUCKETS)).astype(float)
    ref /= ref.sum()

    out = dict(tag=args.tag, tau=tau, dataset="lince" if args.lince else "switchlingua_test",
               overall=overall, per_pair={}, ref_cell_share=ref.tolist())
    for p in sorted(set(pair.tolist())) + ["ALL"]:
        m = np.ones(len(y), bool) if p == "ALL" else pair == p
        yy, pp, pr, bm = y[m], pred[m], prob[m], bucket[m]
        res = dict(f1=prf(yy, pp)["f1"], n=int(m.sum()), switches=int(yy.sum()),
                   std_f1=standardised_f1(yy, pp, cell[m], ref),
                   switch_share_by_bucket={b: float(np.mean(bm[yy] == j)) if yy.any() else 0.0
                                           for j, b in enumerate(BUCKETS)},
                   by_bucket=slice_by(yy, pp, pr, bm, BUCKETS),
                   by_direction=slice_by(yy, pp, pr, direction[m], ["into L2", "into L1"]))
        if cstype is not None:
            cm = cstype[m]
            res["by_cs_type"] = slice_by(yy, pp, pr, cm, ct_names)
            # cs_type validation: where do each type's switches sit?
            res["cs_type_vs_bucket"] = {
                name: {b: float(np.mean(bm[yy & (cm == j)] == bj)) if (yy & (cm == j)).any() else 0.0
                       for bj, b in enumerate(BUCKETS)}
                for j, name in enumerate(ct_names)}
            res["cs_type_x_bucket"] = {
                name: {b: prf(yy[(cm == j) & (bm == bj)], pp[(cm == j) & (bm == bj)])["f1"]
                       for bj, b in enumerate(BUCKETS)} for j, name in enumerate(ct_names)}
        e = ending[m]
        if (e >= 0).any():
            res["by_ending"] = slice_by(yy, pp, pr, e, CATS)
            adj = bm == 0
            res["by_ending_adjacent"] = slice_by(yy[adj], pp[adj], pr[adj], e[adj], CATS)
            res["by_ending_x_bucket"] = {
                b: slice_by(yy[bm == bj], pp[bm == bj], pr[bm == bj], e[bm == bj], CATS)
                for bj, b in enumerate(BUCKETS)}
        out["per_pair"][p] = res

    # duration by cs_type
    if ct_of_row is not None and "dtrue" in z and len(z["dtrue"]):
        dct = ct_of_row.reindex(z["drow"]).fillna(-1).astype(int).to_numpy()
        dpair = np.array(PAIR_KEYS)[z["dpair"]]
        dt, dp = z["dtrue"].astype(int), z["dpred"].astype(int)
        dur = {}
        for p in sorted(set(dpair.tolist())) + ["ALL"]:
            mm = np.ones(len(dt), bool) if p == "ALL" else dpair == p
            dur[p] = {name: {k: v for k, v in duration_scores(dt[mm & (dct == j)], dp[mm & (dct == j)]).items()
                             if k in ("n", "accuracy", "macro_f1", "per_class_f1")}
                      for j, name in enumerate(ct_names)}
            dur[p]["all"] = duration_scores(dt[mm], dp[mm])["macro_f1"]
        out["duration_by_cs_type"] = dur

    # ---- printing
    show = ("adjacent", "comma", "sentence_end", "turn")
    pp_rows = [dict(pair=p, f1=f"{r['f1']:.3f}", std_f1=f"{r['std_f1']:.3f}",
                    adj_share=f"{r['switch_share_by_bucket']['adjacent']:.2f}",
                    **{c: f"{r['by_cs_type'][c]['f1']:.3f}" for c in ct_names if c in r.get("by_cs_type", {})},
                    **{b[:5]: f"{r['by_bucket'][b]['f1']:.2f}" for b in show if b in r["by_bucket"]},
                    adj_fn=f"{r['by_bucket'].get('adjacent', {}).get('share_of_fn', 0):.2f}",
                    into_l2=f"{r['by_direction'].get('into L2', {}).get('f1', 0):.3f}")
               for p, r in sorted(out["per_pair"].items(), key=lambda kv: -kv[1]["f1"])]
    cols = ["pair", "f1", "std_f1", "adj_share", *(ct_names if cstype is not None else []),
            "adjac", "comma", "sente", "turn", "adj_fn", "into_l2"]
    print(fmt_table(pp_rows, cols, "\nPer pair: F1, mix-standardised F1, mid-clause share of switches, "
                                   "F1 by cs_type and gap bucket"))
    if cstype is not None:
        v = out["per_pair"]["ALL"]["cs_type_vs_bucket"]
        print(fmt_table([dict(cs_type=c, **{b: f"{s[b]:.2f}" for b in BUCKETS}) for c, s in v.items()],
                        ["cs_type", *BUCKETS], "\ncs_type validation: share of each type's switches per bucket"))
    for p in ("Korean->English", "Japanese->English", "Chinese->English", "Cantonese->English"):
        r = out["per_pair"].get(p, {})
        for key in ("by_ending", "by_ending_adjacent"):
            if key in r:
                rows_ = [dict(ending=c, n=s["n"], base=f"{s['base_rate']:.3f}", prob=f"{s['mean_prob']:.3f}",
                              f1=f"{s['f1']:.3f}", recall=f"{s['recall']:.3f}", fn_share=f"{s['share_of_fn']:.2f}")
                         for c, s in r[key].items()]
                print(fmt_table(rows_, ["ending", "n", "base", "prob", "f1", "recall", "fn_share"],
                                f"\n{p} {key} (L1 positions)"))
    dest = f"data/results/{'lince_breakdown' if args.lince else 'error_breakdown'}_{args.tag}.json"
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
