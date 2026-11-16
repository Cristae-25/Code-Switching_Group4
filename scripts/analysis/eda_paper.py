#!/usr/bin/env python
"""Corpus statistics for the paper's SwitchLingua section (no model, ~1 min).

    ./.venv/bin/python scripts/analysis/eda_paper.py [--out-fig paper/figures]

A small, reproducible subset of notebooks/switch_eda.ipynb: the numbers the paper
cites, on the final (version-2) word labels in data/final/token_labels_word.parquet,
joined on row_id with the persona and prompt fields of the original SwitchLingua CSVs.

Writes:
    data/results/eda_paper.json   every number the paper's corpus section quotes
    <out-fig>/eda_pairs.png       per pair: corpus share | switch rate by direction

Units are the labelling tokens: words, except Chinese, Japanese, Korean and Cantonese,
which are per character. Punctuation and digits ("-") carry no language and are not
scored, as in the anticipation task.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORD = os.path.join(ROOT, "data/final/token_labels_word.parquet")
CSV_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Shelton1013--SwitchLingua_text/snapshots/*/*.csv")
META = ["age", "gender", "education_level", "topic", "tense", "perspective", "cs_ratio",
        "conversation_type", "cs_type", "cs_function"]
# House style (scripts/plots/plot_pair_f1.py).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, LIGHT = "#2a78d6", "#eb6834", "#b7d3f6"


def pair_name(l1: str, l2: str) -> str:
    l1 = "Tagalog" if l1 == "Philippines" else l1
    return l1 if l2 == "English" and l1 != "German" else f"{l1} → {l2}"


def flatten(w: pd.DataFrame) -> dict:
    n = w.tokens.map(len).to_numpy()
    lang = np.concatenate(w.lang.to_numpy())
    y_sw = np.concatenate(w.y_sw.to_numpy())
    nxt = np.concatenate([lang[1:], np.array(["-"])])
    same_row_next = np.ones(len(lang), bool)
    same_row_next[np.cumsum(n) - 1] = False
    return dict(row=np.repeat(np.arange(len(w)), n), pair=np.repeat(w.pair.to_numpy(), n),
                lang=lang, y_sw=y_sw, y_len=np.concatenate(w.y_len.to_numpy()),
                censored=np.concatenate(w.censored.to_numpy()),
                punct_next=(nxt == "-") & same_row_next)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-fig", default=os.path.join(ROOT, "paper/figures"))
    ap.add_argument("--json", default=os.path.join(ROOT, "data/results/eda_paper.json"))
    args = ap.parse_args()

    w = pd.read_parquet(WORD, columns=["row_id", "first_language", "second_language",
                                       "tokens", "lang", "y_sw", "y_len", "censored"])
    w["pair"] = [pair_name(a, b) for a, b in zip(w.first_language, w.second_language)]
    T = flatten(w)
    scored = T["y_sw"] >= 0
    sw = T["y_sw"] == 1
    word = T["lang"] != "-"
    in_l1, in_l2 = scored & (T["lang"] == "L1"), scored & (T["lang"] == "L2")
    into_l2, into_l1 = sw & (T["lang"] == "L1"), sw & (T["lang"] == "L2")
    closed_l2 = into_l2 & ~T["censored"]           # L2 runs that end inside the text

    g = pd.DataFrame({"pair": T["pair"], "word": word, "scored": scored, "sw": sw,
                      "in_l1": in_l1, "in_l2": in_l2, "into_l2": into_l2, "into_l1": into_l1,
                      "sw_punct": sw & T["punct_next"], "nosw_punct": scored & ~sw & T["punct_next"]})
    per = g.groupby("pair").sum()
    per["conversations"] = w.groupby("pair").size()
    per["switch_rate"] = per.sw / per.scored
    per["p_into_l2"] = per.into_l2 / per.in_l1
    per["p_into_l1"] = per.into_l1 / per.in_l2
    per["l2_share"] = per.in_l2 / (per.in_l1 + per.in_l2)
    per["switch_punct"] = per.sw_punct / per.sw
    per["nonswitch_punct"] = per.nosw_punct / (per.scored - per.sw)
    per["conv_share"] = per.conversations / per.conversations.sum()
    per["word_share"] = per.word / per.word.sum()
    l2len = pd.DataFrame({"pair": T["pair"][closed_l2], "y_len": T["y_len"][closed_l2]})
    per["l2_run_median"] = l2len.groupby("pair").y_len.median()
    per["l2_run_mean"] = l2len.groupby("pair").y_len.mean()
    l1len = pd.DataFrame({"pair": T["pair"][into_l1 & ~T["censored"]],
                          "y_len": T["y_len"][into_l1 & ~T["censored"]]})
    per["l1_run_median"] = l1len.groupby("pair").y_len.median()
    per = per.sort_values("switch_rate")

    # Persona and prompt fields, averaged per conversation (as in the notebook, section 4).
    files = sorted(glob.glob(CSV_GLOB), key=os.path.basename)
    meta = pd.concat([pd.read_csv(f, usecols=META) for f in files], ignore_index=True)
    assert len(meta) == len(w) == 234_172, len(meta)
    c = pd.DataFrame({"row": T["row"], "scored": scored, "sw": sw, "l1": in_l1, "l2": in_l2})
    conv = c.groupby("row")[["scored", "sw", "l1", "l2"]].sum()
    conv["switch_rate"] = conv.sw / conv.scored.clip(lower=1)
    conv["l2_share"] = conv.l2 / (conv.l1 + conv.l2).clip(lower=1)
    rl = pd.DataFrame({"row": T["row"][closed_l2], "y_len": T["y_len"][closed_l2]})
    conv["l2_run"] = rl.groupby("row").y_len.mean()
    conv = conv.join(meta.iloc[w.row_id.to_numpy()].reset_index(drop=True))

    def by(attr):
        t = conv.groupby(attr).agg(conversations=("sw", "size"), switch_rate=("switch_rate", "mean"),
                                   l2_share=("l2_share", "mean"), l2_run=("l2_run", "mean"))
        return {str(k): {kk: (int(vv) if kk == "conversations" else round(float(vv), 4))
                         for kk, vv in r.items()} for k, r in t.iterrows()}

    tot = per[["word", "scored", "sw", "conversations", "into_l2", "into_l1", "sw_punct"]].sum()
    out = {
        "units": "words; per character for Chinese, Japanese, Korean, Cantonese",
        "labels": "data/final/token_labels_word.parquet (version 2)",
        "total": {"conversations": int(tot.conversations), "tokens_all": int(sum(map(len, w.tokens))),
                  "words": int(tot.word), "scored": int(tot.scored), "switches": int(tot.sw),
                  "switch_rate": round(float(tot.sw / tot.scored), 4),
                  "share_into_l2": round(float(tot.into_l2 / tot.sw), 4),
                  "switch_punct": round(float(tot.sw_punct / tot.sw), 4)},
        "per_pair": {p: {k: (int(v) if k in ("conversations", "word", "scored", "sw") else round(float(v), 4))
                         for k, v in r[["conversations", "conv_share", "word", "word_share", "scored", "sw",
                                        "switch_rate", "p_into_l2", "p_into_l1", "l2_share",
                                        "l2_run_median", "l2_run_mean", "l1_run_median",
                                        "switch_punct", "nonswitch_punct"]].items()}
                     for p, r in per.iterrows()},
        "by_attribute": {a: by(a) for a in ["cs_type", "cs_ratio", "conversation_type", "age",
                                            "gender", "education_level", "topic", "cs_function"]},
        "column_values": {a: sorted(map(str, meta[a].dropna().unique())) for a in META},
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=1, ensure_ascii=False)
    print("wrote", args.json)

    # Figure: corpus share (conversations, words) | per-token switch probability by direction.
    plt.rcParams.update({"font.size": 10})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), facecolor=SURFACE, sharey=True,
                               gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(per))
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, colors=INK_2)
    a.barh(y + 0.2, per.conv_share * 100, 0.4, color=LIGHT, label="conversations")
    a.barh(y - 0.2, per.word_share * 100, 0.4, color=BLUE, label="tokens")
    a.set_yticks(y, per.index)
    a.set_xlabel("share of the corpus (%)", color=MUTED)
    a.set_title("Corpus share", loc="left", color=INK, fontsize=11)
    a.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    b.barh(y + 0.2, per.p_into_l2 * 100, 0.4, color=BLUE, label="L1 → L2, per L1 token")
    b.barh(y - 0.2, per.p_into_l1 * 100, 0.4, color=ORANGE, label="L2 → L1, per L2 token")
    b.scatter(per.switch_rate * 100, y, marker="|", s=160, color=INK, zorder=3, label="any switch, per token")
    b.set_xlabel("probability the next token switches (%)", color=MUTED)
    b.set_title("Switch rate by direction", loc="left", color=INK, fontsize=11)
    b.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    os.makedirs(args.out_fig, exist_ok=True)
    path = os.path.join(args.out_fig, "eda_pairs.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", path)

    show = per[["conversations", "word", "switch_rate", "p_into_l2", "p_into_l1", "l2_share",
                "l2_run_median", "l1_run_median", "switch_punct", "nonswitch_punct"]]
    print(show.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(out["total"]))
    for a_ in ("cs_type", "cs_ratio", "age", "conversation_type"):
        print(a_, json.dumps(out["by_attribute"][a_]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
