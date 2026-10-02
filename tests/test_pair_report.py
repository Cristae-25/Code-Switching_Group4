"""One language pair's results page, from result files and saved predictions."""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod():
    spec = importlib.util.spec_from_file_location("pair_report", os.path.join(ROOT, "scripts", "pair_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rep(f1):
    blk = lambda f: dict(f1=f, precision=f, recall=f, n=100, base_rate=0.1)  # noqa: E731
    return dict(overall=blk(f1), per_pair={"German->French": blk(f1), "Spanish->English": blk(0.9)},
                macro_f1=f1, per_direction={"into L2": blk(f1), "into L1": blk(f1)},
                shared=dict(overall=blk(f1), macro_f1=f1, per_pair={"German->French": blk(f1 + 0.1)},
                            per_direction={}))


def _fixtures(tmp_path):
    res = tmp_path / "xlmr.json"
    json.dump(dict(model="xlmr", unit="subword", **_rep(0.6)), open(res, "w"))
    from src.evaluation import PAIR_KEYS
    gf = PAIR_KEYS.index("German->French")
    # one row (id 9), pieces: ich ▁bin ▁müde ▁mais ▁je ; a switch after "müde" (pos 2)
    sw = pd.DataFrame([dict(row_id=9, first_language="German", second_language="French",
                            pieces=["▁ich", "▁bin", "▁müde", "▁mais", "▁je"])])
    swp = tmp_path / "sw.parquet"
    sw.to_parquet(swp, index=False)
    split = dict(logit=np.array([-3.0, -3.0, -4.0, -3.0]), y=np.array([0, 0, 1, 0]),
                 pair=np.full(4, gf), lang=np.array([0, 0, 0, 1]),
                 row=np.full(4, 9), pos=np.array([0, 1, 2, 3]))
    preds = tmp_path / "p.npz"
    np.savez(preds, **{f"{s}_{k}": v for s in ("val", "test") for k, v in split.items()})
    return str(res), str(preds), str(swp)


def test_report_shows_the_pair_in_every_result_file(tmp_path):
    res, preds, swp = _fixtures(tmp_path)
    md = _mod().pair_report("German->French", [res], [], swp)
    assert "German->French" in md and "0.600" in md and "0.700" in md     # switch F1 and shared F1


def test_report_counts_errors_by_direction_and_shows_the_text(tmp_path):
    res, preds, swp = _fixtures(tmp_path)
    md = _mod().pair_report("German->French", [res], [preds], swp, examples=3)
    assert "missed switches" in md.lower()
    assert "▁müde" in md or "müde" in md                                  # the missed switch, in context
    assert "into L2" in md


def test_old_prediction_files_without_positions_are_explained(tmp_path):
    res, preds, swp = _fixtures(tmp_path)
    z = dict(np.load(preds))
    old = tmp_path / "old.npz"
    np.savez(old, **{k: v for k, v in z.items() if not k.endswith(("_row", "_pos"))})
    md = _mod().pair_report("German->French", [res], [str(old)], swp)
    assert "re-run with the current --save-preds" in md
