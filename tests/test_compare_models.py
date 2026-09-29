"""The per-pair comparison table across models."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "compare_models", os.path.join(ROOT, "scripts", "compare_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _report(f1s, into_l2):
    return dict(overall=dict(f1=sum(f1s.values()) / len(f1s)),
                per_pair={k: dict(f1=v) for k, v in f1s.items()},
                macro_f1=sum(f1s.values()) / len(f1s),
                per_direction={"into L2": dict(f1=into_l2), "into L1": dict(f1=0.5)},
                duration=dict(macro_f1=0.4, accuracy=0.6))


def test_table_has_one_row_per_pair_and_ranks_models(tmp_path):
    base = tmp_path / "baseline.json"
    json.dump({"hazard@tuned": _report({"Spanish->English": 0.2, "Hindi->English": 0.4}, 0.0),
               "_meta": {}}, open(base, "w"))
    model = tmp_path / "model.json"
    json.dump(dict(model="xlmr", unit="subword",
                   **_report({"Spanish->English": 0.6, "Hindi->English": 0.5}, 0.45)), open(model, "w"))
    m = _mod()
    runs = m.load_runs([f"{base}:hazard@tuned", str(model)])
    assert [r["label"] for r in runs] == ["hazard@tuned (baseline)", "xlmr (subword)"]
    table = m.pair_table(runs)
    assert list(table.index) == ["Hindi->English", "Spanish->English"]   # hardest pair last
    assert table.loc["Spanish->English", "xlmr (subword)"] == 0.6
    md = m.to_markdown(runs, table)
    assert "into L2" in md and "Spanish->English" in md


def test_zero_shot_run_never_collides_with_the_main_run(tmp_path):
    main, zs = tmp_path / "x.json", tmp_path / "x_zs.json"
    json.dump(dict(model="xlmr", unit="subword", args=dict(train_langs=""),
                   **_report({"Spanish->English": 0.6}, 0.4)), open(main, "w"))
    json.dump(dict(model="xlmr", unit="subword", args=dict(train_langs="Spanish,French"),
                   **_report({"Spanish->English": 0.1}, 0.1),
                   zero_shot=_report({"Hindi->English": 0.3}, 0.2)), open(zs, "w"))
    m = _mod()
    runs = m.load_runs([str(main), str(zs)])
    labels = [r["label"] for r in runs]
    assert len(labels) == len(set(labels)) == 3
    assert m.pair_table(runs).loc["Spanish->English", "xlmr (subword)"] == 0.6
