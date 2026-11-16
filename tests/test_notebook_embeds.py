"""The Kaggle notebooks embed src/ modules verbatim; each must ship everything it imports."""
import ast
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDERS = ["build_pretrained_notebook", "build_untrained_notebook", "build_labels_notebook",
            "build_kaggle_notebook", "build_loss_sweep_notebook"]


def _builder(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "notebooks", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _module_level_src_imports(source: str) -> set[str]:
    """``src`` modules imported at module scope. Imports inside functions are lazy
    (e.g. the model taggers in dispatch.py) and only matter if that code runs."""
    found = set()
    for stmt in ast.parse(source).body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(stmt):
            if not isinstance(n, ast.ImportFrom) or not n.module and n.level != 1:
                continue
            if n.level == 1 and n.module:
                found.add(n.module.split(".")[0])
            elif n.level == 1:
                found |= {a.name for a in n.names}
            elif n.module.startswith("src."):
                found.add(n.module.split(".")[1])
            elif n.module == "src":
                found |= {a.name for a in n.names}
    return found


@pytest.mark.parametrize("name", BUILDERS)
def test_every_embedded_file_ships_what_it_imports(name):
    b = _builder(name)
    shipped = set(b.MODULES)
    files = [os.path.join(ROOT, "src", f"{m}.py") for m in b.MODULES]
    files += [os.path.join(ROOT, "scripts", *f"{s}.py".split("/")) for s in getattr(b, "SCRIPTS", [])]
    missing = {(m, os.path.relpath(f, ROOT)) for f in files
               for m in _module_level_src_imports(open(f).read()) if m not in shipped}
    assert not missing, f"{name}.MODULES lacks: {sorted(missing)}"


@pytest.mark.parametrize("name", ["build_pretrained_notebook", "build_untrained_notebook",
                                  "build_loss_sweep_notebook"])
def test_the_training_notebooks_run_their_training_script(tmp_path, name):
    """Write the notebook's %%writefile cells to a clean folder and import the
    training script from there, as Kaggle does: a module left out fails here."""
    out = tmp_path / "nb.ipynb"
    b = _builder(name)
    b.OUT = str(out)
    if hasattr(b, "main"):
        argv, sys.argv = sys.argv, ["build"]
        try:
            b.main()
        finally:
            sys.argv = argv
    nb = json.load(open(out))
    work = tmp_path / "work"
    for d in ("src", "scripts"):
        (work / d).mkdir(parents=True)
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and src.startswith("%%writefile "):
            head, body = src.split("\n", 1)
            dest = work / head.split(" ", 1)[1].strip()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body)
    r = subprocess.run([sys.executable, "scripts/training/train_anticipation.py", "--help"], cwd=work,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--freeze-backbone" in r.stdout and "--dur-classes" in r.stdout
