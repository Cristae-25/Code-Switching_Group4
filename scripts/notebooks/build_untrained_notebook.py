#!/usr/bin/env python
"""Write kaggle/untrained_xlmr.ipynb: how well does XLM-R do without fine-tuning?

    ./.venv/bin/python scripts/notebooks/build_untrained_notebook.py

Two runs on the 6 script-disjoint pairs, both with the backbone exactly as
pretrained (``--freeze-backbone``):

    xlmr_untrained_6pairs  nothing trained: random heads, only the switch threshold
                           and duration offsets are fit (on val)
    xlmr_probe_6pairs      the two linear heads trained, the backbone frozen

Same labels, splits, causal mask and report as the fine-tuned ``xlmr_6pairs``
run, so the three compare position for position. It reuses the setup, project
code and label cells of ``build_pretrained_notebook.py``; re-run this script
after changing src/ or the training script.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_pretrained_notebook import DATA, MODULES, ROOT, SCRIPTS, SETUP, code, md  # noqa: E402,F401

OUT = os.path.join(ROOT, "kaggle", "untrained_xlmr.ipynb")

INTRO = """
# XLM-R without fine-tuning: anticipate code switches and their duration

How much of the fine-tuned result (`xlmr_6pairs`: switch macro-F1 0.703, duration macro-F1 0.602) comes from what XLM-R already knows, and how much from fine-tuning? This notebook runs pretrained XLM-R **with its weights left exactly as downloaded**, on the same 6 script-disjoint pairs (Cantonese, Chinese, Japanese, Korean, Russian, Arabic → English), with the same causal mask, labels, splits and report.

XLM-R has no "switch" or "duration" output of its own, so something has to turn its features into the two predictions. Two runs show the two ends:

| Run | Backbone | Heads (one linear layer each) | What it measures |
|---|---|---|---|
| `xlmr_untrained_6pairs` | pretrained, frozen | **random, never trained** | the literal untrained model. Only the switch threshold and the 3 duration offsets are fit, on val. Expect chance-level scores |
| `xlmr_probe_6pairs` | pretrained, frozen | **trained** (12,000 steps, as the fine-tuned run) | a *linear probe*: how much switch and duration information is already in the pretrained features, readable by a single linear layer |

- **Fine-tuned − probe** is the gain from adapting the backbone itself.
- **Probe − untrained** is how much the pretrained features carry on their own.

The 9 Latin-script pairs are scored zero-shot in both runs, as in `xlmr_6pairs`.

| Split | Share | Used for |
|---|---|---|
| train | 80% | fitting the probe's heads (`row_id % 10` in 0–7; unused by the untrained run) |
| val | 10% | the switch threshold and duration offsets (8) |
| test | 10% | the final numbers, scored once (9) |

## How to run

1. **Settings:** set the Accelerator to **GPU T4 ×2** (one GPU is used) and turn Internet **on**.
2. **Secret:** Add-ons → Secrets → `HF_TOKEN`, a read token with access to `CodeSwitchGroup4/switchlingua-tagged-final` (a fine-grained token needs the `CodeSwitchGroup4` org under its permissions) and the gated `Shelton1013/SwitchLingua_text`.
3. **Save Version → Save & Run All.** Use `SMOKE = True` first for a quick check. Expect roughly 15–25 minutes for the untrained run (evaluation only) and 35–50 for the probe: the frozen backbone needs no backward pass.
4. **Collect the results:** download `data/results/anticipation_xlmr_untrained_6pairs.json`, `data/results/anticipation_xlmr_probe_6pairs.json` and the two `data/preds/*.npz` from the Output tab into the same paths in the repo.
"""

CONFIG = """
MODEL = "xlmr"          # the backbone (the label cell builds this tokenizer's pieces)
SMOKE = False           # True: 60 probe steps + a small evaluation, to check everything runs
RUN_UNTRAINED = True    # random heads, nothing trained
RUN_PROBE = True        # frozen backbone, heads trained
PROBE_STEPS = 12_000    # the same number of steps as the fine-tuned xlmr_6pairs
BATCH = 16
WINDOW, STRIDE = 384, 288
HEAD_LR = 1e-3          # the heads' learning rate (the backbone gets none)
EVAL_BATCHES = 0        # 0 = score the whole val and test splits
SEED = 0                # the random heads' initialisation, and data order
PAIRS = "Cantonese,Chinese,Japanese,Korean,Russian,Arabic"   # "" = all 15 pairs
LABELS_REPO = "CodeSwitchGroup4/switchlingua-tagged-final"
"""

RUN = """
probe_steps, evalb = (60, 20) if SMOKE else (PROBE_STEPS, EVAL_BATCHES)
common = (f"python scripts/training/train_anticipation.py --model {MODEL} --unit subword --freeze-backbone "
          f"--batch-size {BATCH} --window {WINDOW} --stride {STRIDE} --head-lr {HEAD_LR} "
          f"--eval-batches {evalb} --workers 2 --seed {SEED}" + (f" --train-langs {PAIRS}" if PAIRS else ""))
suffix = f"_{len(PAIRS.split(','))}pairs" if PAIRS else ""     # no suffix = all 15, as in train_pretrained.ipynb
RUNS = {f"{MODEL}_untrained{suffix}": (RUN_UNTRAINED, 0), f"{MODEL}_probe{suffix}": (RUN_PROBE, probe_steps)}
for tag, (on, steps) in RUNS.items():
    if not on:
        print(f"{tag}: off")
        continue
    cmd = (f"{common} --steps {steps} --out data/results/anticipation_{tag}.json "
           f"--ckpt models/anticipation_{tag}.pt --save-preds data/preds/{tag}.npz")
    print(cmd)
    get_ipython().system(f"{cmd} 2>&1 | tee logs/train_{tag}.log")
"""

RESULTS = """
import json
# The fine-tuned reference, xlmr_6pairs (docs/results/2026-10-12_xlmr_6pairs_duration.md).
FT = {"switch_f1": {"Korean->English": 0.813, "Japanese->English": 0.741, "Chinese->English": 0.702,
                    "Arabic->English": 0.669, "Russian->English": 0.666, "Cantonese->English": 0.625},
      "dur_f1": {"Korean->English": 0.597, "Japanese->English": 0.623, "Chinese->English": 0.607,
                 "Arabic->English": 0.596, "Russian->English": 0.593, "Cantonese->English": 0.598},
      "macro": 0.703, "dur_macro": 0.602, "zero_shot": 0.514}
if PAIRS != "Cantonese,Chinese,Japanese,Korean,Russian,Arabic":
    FT = None                                  # the reference above is for those 6 pairs only
ref = lambda key: f" | fine-tuned {FT[key]:.3f}" if FT else ""
cols = {}
for tag in RUNS:
    f = f"data/results/anticipation_{tag}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    o, t = r["overall"], r.get("duration_tuned", {}).get("report", {})
    td = t.get("duration", r["duration"])
    print(f"== {tag}: {r['steps']} steps, tau {r['tau']:.3f}")
    print(f"  switch F1 {o['f1']:.3f} (P {o['precision']:.3f}, R {o['recall']:.3f}) | macro-F1 {r['macro_f1']:.3f}"
          + ref("macro"))
    print("  by direction:", {k: round(v["f1"], 3) for k, v in r["per_direction"].items()})
    print(f"  duration (val-tuned): accuracy {td['accuracy']:.3f} | macro-F1 {td['macro_f1']:.3f} | "
          "F1 small/medium/large " + " / ".join(f"{x:.3f}" for x in td["per_class_f1"])
          + ref("dur_macro"))
    if "zero_shot" in r:
        print(f"  zero-shot ({len(r['zero_shot']['per_pair'])} held-out pairs) F1 "
              f"{r['zero_shot']['overall']['f1']:.3f}" + ref("zero_shot"))
    dpp = t.get("duration_per_pair", r["duration_per_pair"])
    name = tag.split("_")[1]
    cols[f"switch F1, {name}"] = {k: v["f1"] for k, v in r["per_pair"].items()}
    cols[f"duration F1, {name}"] = {k: v["macro_f1"] for k, v in dpp.items()}
if FT:
    cols["switch F1, fine-tuned"], cols["duration F1, fine-tuned"] = FT["switch_f1"], FT["dur_f1"]
table = pd.DataFrame(cols)
display(table.sort_values(table.columns[-2 if FT else 0], ascending=False).round(3))
"""

CLEANUP = """
for tag in RUNS:                               # an untrained or probe checkpoint isn't worth keeping
    if os.path.exists(f"models/anticipation_{tag}.pt"):
        os.remove(f"models/anticipation_{tag}.pt")
if os.path.exists(PATH):                       # derived from the private labels: don't publish it
    os.remove(PATH)
print(sorted(os.listdir("data")), sorted(os.listdir("models")))
"""


def main() -> int:
    cells = [md(INTRO), md("## Config"), code(CONFIG), md("## Setup"), code(SETUP),
             md("## Project code\nThe live `src/` modules and the training script, embedded verbatim.")]
    for name in MODULES:
        body = open(os.path.join(ROOT, "src", f"{name}.py")).read() or "# package marker\n"
        cells.append(code(f"%%writefile src/{name}.py\n{body}"))
    for name in SCRIPTS:
        body = open(os.path.join(ROOT, "scripts", *f"{name}.py".split("/"))).read()
        cells.append(code(f"%%writefile scripts/{name}.py\n{body}"))
    cells += [md("## Build the labels for this tokenizer"), code(DATA),
              md("## Run: untrained heads, then the linear probe (backbone frozen in both)"), code(RUN),
              md("## Results, next to the fine-tuned model"), code(RESULTS),
              md("## Clean up\nRemoves the checkpoints and the derived training file. Result JSONs stay in `data/results/`."),
              code(CLEANUP)]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({len(cells)} cells, {len(MODULES)} modules, {len(SCRIPTS)} script)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
