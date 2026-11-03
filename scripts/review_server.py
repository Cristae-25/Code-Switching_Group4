#!/usr/bin/env python
"""Local web tool for spot-checking token language labels.

    ./.venv/bin/python scripts/review_server.py            # serve on :8000 (data/tagged.parquet)
    ./.venv/bin/python scripts/review_server.py --report   # what you corrected

A *spot-check*, not an annotation effort: accuracy numbers come from
``eval_public.py`` on human-annotated public data. This exists to catch
systematic errors by eye - the kind that found the "Apple is English" and
Devanagari bugs - so the sample is deliberately aimed at likely mistakes:

* 15 rows per language pair (30 for Hindi and Malay, the lowest-agreement
  pairs), drawn from a seeded pool and ranked by how many named entities they
  carry plus how many tokens AnE-LID and dict-freq disagree on. A third of
  each pair is left random so the review is not *only* hard cases.
* tokens where the two taggers disagree are underlined in the UI.

Click a token to cycle L1 -> L2 -> other. Names are shown in italics with a
green outline (the ``is_entity`` flag); like every word they carry a language -
check that it is the right one ("Game of Thrones" English, "Messi" the
language of its clause). Verdicts land in
data/review.sqlite (resumable). ``--report`` groups corrections by type
(e.g. ``NE -> L2``) with the actual tokens, which is what turns a spot-check
into a code fix. Nothing is ever written back to the labels - fix the pattern
in code and re-tag instead, so the pipeline stays fully automated.

Standard library only apart from pandas (already needed to read the parquet) -
no Flask, no node, nothing new to install.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PARQUET = os.path.join(DATA, "tagged.parquet")         # final labels (see CLAUDE.md)
BOOST = {"Hindi": 2, "Malay": 2}                        # lowest AnE/dict agreement
DB = os.path.join(DATA, "review.sqlite")

STATE: dict = {}


# --------------------------------------------------------------------------- db
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
        row_id INTEGER PRIMARY KEY,
        first_language TEXT, second_language TEXT, quality TEXT, tier TEXT,
        n_tokens INTEGER, n_changed INTEGER,
        gold_labels TEXT, predicted_labels TEXT, reviewed_at TEXT,
        verified INTEGER DEFAULT 1)""")
    # tolerate databases created before `verified` existed
    cols = {r[1] for r in conn.execute("PRAGMA table_info(reviews)")}
    if "verified" not in cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN verified INTEGER DEFAULT 1")
    conn.commit()
    return conn


def _disagreements(tokens, labels, l1, l2) -> list[int]:
    """Indices where dict-freq names a different language than the stored label."""
    from src.dict_tagger import tag_frequency
    out = []
    for k, ((_, d), lab) in enumerate(zip(tag_frequency(list(tokens), l1, l2), labels)):
        if d in (l1, l2) and lab in (l1, l2) and d != lab:
            out.append(k)
    return out


def counted_labels(tokens, token_lang, is_entity, first_language: str,
                   second_language: str) -> list[str]:
    """Each token's language as the training labels count it.

    ``token_lang`` is the tagger's raw label; for names, the default names
    policy (``segments.resolve_names``) decides what is counted, so showing
    the raw label made correct names look wrong (Asia/Milan stored L1 but
    counted English). Names the policy leaves ``other`` (Elanga) take the
    language ``absorb_other`` folds them into. Punctuation stays ``other``.
    """
    sys.path.insert(0, ROOT)
    from src.segments import DEFAULT_NAMES, absorb_other, resolve_names
    ent = [bool(x) for x in is_entity]
    tagged = resolve_names(list(zip(tokens, token_lang)), ent, first_language,
                           second_language, DEFAULT_NAMES)
    absorbed = absorb_other(tagged)
    return [a if e and l == "other" and any(c.isalpha() for c in t) else l
            for (t, l), (_, a), e in zip(tagged, absorbed, ent)]


def _counted(r) -> list[str]:
    if "is_entity" not in r.index:                   # pre-policy parquet: names are "NE"
        return list(r.token_lang)
    return counted_labels(list(r.tokens), list(r.token_lang), list(r.is_entity),
                          r.first_language, r.second_language)


def load_override_sample(per_pair: int, seed: int, languages=None):
    """``--overrides``: check the lexical override (src/lexical_override.py).

    Exactly ``per_pair`` rows per pair (no boost): two thirds drawn at random
    from rows where the override relabelled at least one token, the rest at
    random from the pair as a control. Corrected tokens are outlined in the UI.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, ROOT)
    frame = pd.read_parquet(PARQUET)
    if "lexical_override" not in frame.columns:
        raise SystemExit(f"{PARQUET} has no lexical_override column - rebuild with "
                         "scripts/apply_names_policy.py (see CLAUDE.md).")
    if languages:
        frame = frame[frame.first_language.isin(languages)]
    has = frame.lexical_override.map(lambda m: bool(np.any(m)))
    parts = []
    for (l1, l2), g in frame.groupby(["first_language", "second_language"]):
        flipped = g[has.loc[g.index]]
        n_flip = min(len(flipped), (2 * per_pair) // 3) if len(flipped) else 0
        a = flipped.sample(n=n_flip, random_state=seed) if n_flip else flipped.iloc[:0]
        rest = g[~g.row_id.isin(a.row_id)]
        b = rest.sample(n=min(per_pair - n_flip, len(rest)), random_state=seed)
        parts.append(pd.concat([a, b]))
    return (pd.concat(parts).sort_values(["first_language", "second_language", "row_id"])
              .reset_index(drop=True))


def load_sample(per_pair: int, seed: int, tiers=None, qualities=None,
                languages=None, control: int = 0, pool: int = 200):
    """Deterministic sample aimed at likely errors, per language pair.

    For each pair: draw a seeded pool, score rows by NE count + 2x AnE/dict
    disagreements (only for ``ane`` rows - elsewhere the stored label *is*
    dict-freq or exact), keep the top two thirds and fill the rest at random.
    ``control`` adds N exact unicode rows per pair: if you "correct" those,
    the error is in the review.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, ROOT)
    if not os.path.exists(PARQUET):
        raise SystemExit(f"{PARQUET} not found - build the final labels first "
                         "(scripts/apply_names_policy.py, see CLAUDE.md).")
    frame = pd.read_parquet(PARQUET)

    kept = frame
    if tiers:
        kept = kept[kept.method.isin(tiers)]
    if qualities:
        kept = kept[kept.quality.isin(qualities)]
    if languages:
        kept = kept[kept.first_language.isin(languages)]
    if kept.empty:
        raise SystemExit("No rows match those filters. Try --list to see options.")

    rng = np.random.default_rng(seed)
    parts = []
    for (l1, l2), g in kept.groupby(["first_language", "second_language"]):
        n = per_pair * BOOST.get(l1, 1)
        cand = g.sample(n=min(pool, len(g)), random_state=seed)
        score = []
        for r in cand.itertuples():
            ne = (sum(map(bool, r.is_entity)) if hasattr(r, "is_entity")
                  else sum(1 for x in r.token_lang if x == "NE"))
            dis = len(_disagreements(r.tokens, r.token_lang, l1, l2)) if r.method == "ane" else 0
            score.append(ne + 2 * dis)
        cand = cand.assign(_score=score).sort_values("_score", ascending=False)
        hard = cand.head((2 * n) // 3)
        rest = cand.iloc[len(hard):]
        easy = rest.iloc[rng.permutation(len(rest))[:n - len(hard)]]
        parts.append(pd.concat([hard, easy]))
    sample = pd.concat(parts)

    if control and tiers and "unicode" not in tiers:
        spare = frame[(frame.method == "unicode") & (~frame.row_id.isin(sample.row_id))]
        if not spare.empty:
            parts.append(spare.groupby("first_language").sample(
                n=min(control, int(spare.first_language.value_counts().min())),
                random_state=seed))
            sample = pd.concat(parts)

    return (sample.drop(columns="_score", errors="ignore")
                  .sort_values(["first_language", "second_language", "row_id"])
                  .reset_index(drop=True))


def row_payload(i: int) -> dict:
    s = STATE["sample"]
    r = s.iloc[i]
    conn = db()
    saved = conn.execute(
        "SELECT gold_labels, reviewed_at, n_changed FROM reviews WHERE row_id=?",
        (int(r.row_id),)).fetchone()
    conn.close()
    predicted = _counted(r)
    flags = (_disagreements(r.tokens, predicted, r.first_language, r.second_language)
             if r.get("method") == "ane" else [])
    ents = ([bool(x) for x in r.is_entity] if "is_entity" in r.index
            else [x == "NE" for x in predicted])
    ovr = ([k for k, x in enumerate(r.lexical_override) if x]
           if "lexical_override" in r.index else [])
    return dict(
        index=i, total=len(s), row_id=int(r.row_id), flags=flags, entities=ents,
        overridden=ovr,
        first_language=r.first_language, second_language=r.second_language,
        quality=r.quality, tier=str(r.get("method", "")),
        tokens=list(r.tokens),
        predicted=predicted,
        labels=json.loads(saved[0]) if saved else predicted,
        verified=saved is not None,
        verified_at=saved[1] if saved else None,
        n_changed=saved[2] if saved else 0)


def save_verdict(payload: dict) -> dict:
    import datetime
    s = STATE["sample"]
    r = s[s.row_id == payload["row_id"]].iloc[0]
    predicted = _counted(r)
    gold = payload["labels"]
    changed = sum(1 for a, b in zip(predicted, gold) if a != b)
    conn = db()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    conn.execute("""INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (
        int(r.row_id), r.first_language, r.second_language, r.quality,
        str(r.get("method", "")), len(predicted), changed,
        json.dumps(gold), json.dumps(predicted), stamp, 1))
    conn.commit()
    conn.close()
    return dict(ok=True, changed=changed, verified=True, verified_at=stamp)


def progress() -> dict:
    """Reviewed count for the *current* sample, so the bar tracks this session."""
    conn = db()
    if "sample" in STATE:
        ids = [int(x) for x in STATE["sample"].row_id]
        marks = ",".join("?" * len(ids))
        done = conn.execute(
            f"SELECT COUNT(*) FROM reviews WHERE row_id IN ({marks})", ids).fetchone()[0]
        total = len(ids)
    else:
        done = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        total = done
    conn.close()
    return dict(reviewed=done, total=total)


def report(tiers=None, qualities=None, languages=None):
    """Per-pair accuracy on reviewed rows, plus every correction grouped by type."""
    conn = db()
    rows = conn.execute("""SELECT row_id, first_language, second_language, quality, tier,
                                  gold_labels, predicted_labels FROM reviews""").fetchall()
    conn.close()
    if tiers:
        rows = [r for r in rows if r[4] in tiers]
    if qualities:
        rows = [r for r in rows if r[3] in qualities]
    if languages:
        rows = [r for r in rows if r[1] in languages]

    sys.path.insert(0, ROOT)
    import pandas as pd
    tok = {}
    if rows and os.path.exists(PARQUET):
        f = pd.read_parquet(PARQUET, columns=["row_id", "tokens"])
        ids = {r[0] for r in rows}
        tok = {int(a): list(b) for a, b in zip(f.row_id, f.tokens) if int(a) in ids}

    def code(lab, l1, l2):
        return "L1" if lab == l1 else "L2" if lab == l2 else "NE" if lab == "NE" else "other"

    agg: dict[str, dict] = {}
    kinds: dict[str, list] = {}
    for rid, l1, l2, quality, tier, gold, pred in rows:
        gold, pred = json.loads(gold), json.loads(pred)
        pair = f"{l1}->{l2}"
        a = agg.setdefault(pair, dict(pair=pair, quality=quality, rows=0, tokens=0, wrong=0))
        a["rows"] += 1
        toks = tok.get(rid, [""] * len(pred))
        for k, (p, g) in enumerate(zip(pred, gold)):
            if not any(c.isalpha() for c in toks[k]) and toks[k]:
                continue                      # punctuation is not scored
            a["tokens"] += 1
            if p != g:
                a["wrong"] += 1
                kinds.setdefault(f"{code(p, l1, l2)} -> {code(g, l1, l2)}", []).append(
                    (pair, toks[k], rid))
    out = []
    for a in agg.values():
        n, w = a["tokens"], a["wrong"]
        acc = 1 - w / n if n else float("nan")
        z = 1.96
        if n:
            d = 1 + z * z / n
            c = (acc + z * z / (2 * n)) / d
            m = z * ((acc * (1 - acc) / n + z * z / (4 * n * n)) ** 0.5) / d
        else:
            c = m = float("nan")
        out.append({**a, "accuracy": acc, "ci_low": c - m, "ci_high": c + m})
    return sorted(out, key=lambda r: r["accuracy"]), kinds


# ------------------------------------------------------------------------- html
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Code-switch label review</title><style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#b5b4ae;
--l1:#12467f;--l1bg:#dce9f9;--l2:#8a3512;--l2bg:#fce3d8;--ne:#3d5a1e;--nebg:#e2efd3;
--other:#8a8a85;--otherbg:#eeedea;}
@media(prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--l1:#bcd7f7;--l1bg:#1b3554;--l2:#f6c3ac;--l2bg:#5a2a12;--ne:#cfe6b4;--nebg:#2f4419;
--other:#9a9992;--otherbg:#2c2c2a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--muted);
padding:12px 22px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;z-index:10}
h1{font-size:15px;margin:0;font-weight:600}
.bar{flex:1;height:6px;background:var(--otherbg);border-radius:3px;overflow:hidden;min-width:120px}
.bar>div{height:100%;background:#2a78d6;transition:width .2s}
main{padding:22px;max-width:1000px;margin:0 auto}
.meta{color:var(--ink2);font-size:13px;margin-bottom:14px}
.badge{border-radius:3px;padding:1px 7px;font-size:11px;color:#fff;margin-left:6px}
.tok{display:inline-block;margin:2px 1px;padding:3px 7px;border-radius:4px;cursor:pointer;
border:1px solid transparent;user-select:none}
.tok:hover{border-color:var(--ink2)}
.tok.L1{background:var(--l1bg);color:var(--l1)}
.tok.L2{background:var(--l2bg);color:var(--l2);font-weight:600}
.tok.other{background:var(--otherbg);color:var(--other)}
.tok.NE{background:var(--nebg);color:var(--ne);font-style:italic}
.tok.ent{font-style:italic;box-shadow:inset 0 0 0 2px var(--ne)}
.tok.flag{text-decoration:underline dotted 2px;text-underline-offset:4px}
.tok.changed{outline:2px solid #eda100;outline-offset:1px}
.tok.ovr{outline:2px dashed #6250d6;outline-offset:1px}
.sub{font-size:9px;opacity:.65;display:block;text-align:center;letter-spacing:.04em}
button{font:inherit;padding:7px 15px;border-radius:6px;border:1px solid var(--muted);
background:transparent;color:var(--ink);cursor:pointer}
button.primary{background:#2a78d6;color:#fff;border-color:#2a78d6}
button:disabled{opacity:.4;cursor:default}
kbd{background:var(--otherbg);border-radius:3px;padding:1px 5px;font-size:11px;color:var(--ink2)}
.legend{font-size:12px;color:var(--ink2);margin:16px 0 8px}
#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#1baf7a;
color:#fff;padding:8px 16px;border-radius:6px;opacity:0;transition:opacity .25s;font-size:13px}
.verified{background:#1baf7a;color:#fff;border-radius:4px;padding:3px 10px;font-size:12px;
font-weight:600;letter-spacing:.02em}
.unverified{background:var(--otherbg);color:var(--ink2);border-radius:4px;padding:3px 10px;
font-size:12px}
#vbar{margin:10px 0 4px;font-size:12px;color:var(--ink2)}
</style></head><body>
<header>
  <h1>Token label review</h1>
  <div class="bar"><div id="prog" style="width:0%"></div></div>
  <span id="count" class="meta" style="margin:0"></span>
  <button id="prev">&larr;</button><button id="next">&rarr;</button>
  <button id="save" class="primary">Save &amp; next</button>
</header>
<main>
  <div class="meta" id="meta"></div>
  <div id="vbar"></div>
  <div class="legend">Click a token to cycle L1 &rarr; L2 &rarr; other &middot;
    <i style="box-shadow:inset 0 0 0 2px var(--ne);padding:0 3px">outlined italics</i> = named entity, shown as the language it
    <b>counts as</b> under the default names policy (check that language) &middot;
    <u style="text-decoration-style:dotted">dotted underline</u> = AnE and dict-freq disagree &middot;
    <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &middot; <kbd>Enter</kbd> save &amp; next &middot;
    <kbd>r</kbd> reset row &middot; gold outline = you changed it &middot;
    <span style="outline:2px dashed #6250d6;padding:0 3px">violet dashed</span> = relabelled by the lexical override
    (AnE had the other language)</div>
  <div id="toks"></div>
</main>
<div id="toast"></div>
<script>
let S=null, i=0;
const CYCLE=(d)=>d.predicted.includes("NE")
  ? [d.first_language, d.second_language, "NE", "other"]      // pre-policy parquet
  : [d.first_language, d.second_language, "other"];
function cls(l){ if(l===S.first_language) return "L1"; if(l===S.second_language) return "L2"; if(l==="NE") return "NE"; return "other"; }
function short(l){ if(l===S.first_language) return "L1"; if(l===S.second_language) return "L2"; if(l==="NE") return "NE"; return "·"; }
async function load(n){
  S = await (await fetch("/api/row?i="+n)).json(); i = S.index; render();
  const p = await (await fetch("/api/progress")).json();
  document.getElementById("prog").style.width = (100*p.reviewed/p.total)+"%";
  document.getElementById("count").textContent = `${p.reviewed}/${p.total} reviewed · row ${i+1} of ${S.total}`;
}
function render(){
  const q={exact:"#1baf7a",provisional:"#eda100",low:"#e34948"}[S.quality]||"#8a8a85";
  document.getElementById("meta").innerHTML =
    `<b>${S.first_language} &rarr; ${S.second_language}</b>`
    + ((S.overridden||[]).length?` &middot; ${S.overridden.length} token(s) relabelled by the override`:``)
    + `<span class="badge" style="background:${q}">${S.quality}</span>`
    + ` &nbsp; row_id ${S.row_id} &middot; ${S.tokens.length} tokens &middot; auto-tagged by ${S.tier||"?"}`;
  document.getElementById("vbar").innerHTML = S.verified
    ? `<span class="verified">&#10003; REVIEWED</span> `
      + `<span style="margin-left:8px">saved ${S.verified_at}`
      + (S.n_changed?` &middot; you corrected ${S.n_changed} token(s)`:` &middot; no corrections needed`)
      + `</span>`
    : `<span class="unverified">not yet reviewed</span>`
      + `<span style="margin-left:8px">click tokens to correct, then Save</span>`;
  const box=document.getElementById("toks"); box.innerHTML="";
  S.tokens.forEach((t,k)=>{
    const e=document.createElement("span");
    e.className="tok "+cls(S.labels[k])+(S.labels[k]!==S.predicted[k]?" changed":"")
      +(S.flags.includes(k)?" flag":"")+(S.entities[k]?" ent":"")
      +((S.overridden||[]).includes(k)&&S.labels[k]===S.predicted[k]?" ovr":"");
    e.innerHTML=`${t.replace(/[<&]/g,c=>({'<':'&lt;','&':'&amp;'}[c]))}<span class="sub">${short(S.labels[k])}</span>`;
    e.onclick=()=>{const c=CYCLE(S);S.labels[k]=c[(c.indexOf(S.labels[k])+1)%c.length];render();};
    box.appendChild(e);
  });
}
function toast(m){const t=document.getElementById("toast");t.textContent=m;t.style.opacity=1;
  setTimeout(()=>t.style.opacity=0,1200);}
async function save(){
  const r=await (await fetch("/api/verdict",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({row_id:S.row_id,labels:S.labels})})).json();
  toast(r.changed?`saved — ${r.changed} correction(s)`:"saved — no changes");
  if(i+1<S.total) load(i+1); else load(i);
}
document.getElementById("prev").onclick=()=>load(Math.max(0,i-1));
document.getElementById("next").onclick=()=>load(Math.min(S.total-1,i+1));
document.getElementById("save").onclick=save;
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")load(Math.max(0,i-1));
  else if(e.key==="ArrowRight")load(Math.min(S.total-1,i+1));
  else if(e.key==="Enter")save();
  else if(e.key==="r"){S.labels=[...S.predicted];render();}
});
load(0);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        parts = urlparse(self.path)
        if parts.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if parts.path == "/api/row":
            i = int(parse_qs(parts.query).get("i", ["0"])[0])
            i = max(0, min(i, len(STATE["sample"]) - 1))
            return self._send(json.dumps(row_payload(i)).encode(), "application/json")
        if parts.path == "/api/progress":
            return self._send(json.dumps(progress()).encode(), "application/json")
        if parts.path == "/api/report":
            return self._send(json.dumps(report()[0]).encode(), "application/json")
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/verdict":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        self._send(json.dumps(save_verdict(payload)).encode(), "application/json")


def print_report(tiers=None, qualities=None, languages=None) -> None:
    rows, kinds = report(tiers, qualities, languages)
    if not rows:
        print("No reviewed rows match those filters yet.")
        return
    print("Word-token accuracy on reviewed rows (punctuation not scored). A spot-check:")
    print("the rows were chosen to be hard, so this understates overall accuracy.\n")
    print(f"{'pair':22}{'rows':>6}{'words':>8}{'fixed':>7}{'agreed':>9}  95% CI")
    for r in rows:
        print(f"{r['pair']:22}{r['rows']:>6}{r['tokens']:>8}{r['wrong']:>7}"
              f"{r['accuracy']:>8.1%}   [{r['ci_low']:.1%}, {r['ci_high']:.1%}]")
    tot, bad = sum(r["tokens"] for r in rows), sum(r["wrong"] for r in rows)
    print(f"{'ALL':22}{sum(r['rows'] for r in rows):>6}{tot:>8}{bad:>7}{1 - bad / tot:>8.1%}")

    if kinds:
        print("\nCorrections by type (label given -> your label), most common first:")
        for kind, items in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            from collections import Counter
            words = Counter(t for _, t, _ in items).most_common(12)
            print(f"  {kind:12} x{len(items):<4} " + ", ".join(
                f"{w}" + (f" x{c}" if c > 1 else "") for w, c in words))


def main() -> int:
    global PARQUET
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", type=int, default=15,
                    help="rows per language pair (x2 for Hindi and Malay)")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--hard", action="store_true",
                    help="only model-tagged rows (ane/dict); skips the unicode "
                         "rows, which are exact by construction apart from NE")
    ap.add_argument("--tier", default=None,
                    help="comma list: unicode,ane,dict")
    ap.add_argument("--quality", default=None, help="comma list: exact,provisional")
    ap.add_argument("--lang", default=None, help="comma list of first_language")
    ap.add_argument("--control", type=int, default=0,
                    help="also include N unicode rows/language as a control")
    ap.add_argument("--list", action="store_true", help="show filter options")
    ap.add_argument("--overrides", action="store_true",
                    help="check the lexical override: exactly --per-pair rows per pair, "
                         "2/3 with relabelled tokens (outlined), 1/3 random")
    args = ap.parse_args()
    PARQUET = args.parquet

    csv = lambda v: [x.strip() for x in v.split(",")] if v else None
    tiers, qualities = csv(args.tier), csv(args.quality)
    languages = csv(args.lang)
    if args.hard and not tiers:
        tiers = ["ane", "dict"]

    os.makedirs(DATA, exist_ok=True)

    if args.list:
        import pandas as pd
        f = pd.read_parquet(PARQUET)
        print(f.groupby(["method", "quality", "first_language"]).size().to_string())
        return 0

    if args.report:
        print_report(tiers, qualities, languages)
        return 0

    print("building the review sample (scoring candidate rows) ...", flush=True)
    STATE["sample"] = (load_override_sample(args.per_pair, args.seed, languages)
                       if args.overrides else
                       load_sample(args.per_pair, args.seed, tiers,
                                   qualities, languages, args.control))
    done = progress()
    s = STATE["sample"]
    print(f"review set: {len(s)} rows ({args.per_pair}/pair"
          f"{'' if args.overrides else ', x2 for Hindi and Malay'}; "
          f"seed={args.seed}) from {PARQUET}")
    if tiers or qualities or languages:
        print(f"filters: tier={tiers or 'all'} quality={qualities or 'all'} "
              f"lang={languages or 'all'}")
    print("  " + s.groupby(["method", "quality"]).size().to_string().replace("\n", "\n  "))
    print(f"tokens to review: {s.tokens.map(len).sum():,}")
    print(f"already reviewed (all sets): {done['reviewed']}")
    print(f"\n  ->  http://localhost:{args.port}\n\nCtrl-C to stop.")
    if not args.no_open:
        webbrowser.open(f"http://localhost:{args.port}")
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. Progress is saved in data/review.sqlite")
        print("Run with --report to see your corrections grouped by type.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
