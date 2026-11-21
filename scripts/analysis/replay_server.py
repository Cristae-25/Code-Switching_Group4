#!/usr/bin/env python
"""Local web dashboard: replay XLM-R / mBERT anticipation predictions on random rows.

    ./.venv/bin/python scripts/analysis/replay_server.py                 # http://localhost:8001
    ./.venv/bin/python scripts/analysis/replay_server.py --run roberta=roberta_gap --no-open
    ./.venv/bin/python scripts/analysis/replay_server.py --dataset lince  # LinCE tweets, gold tags

Pick a model (or both, side by side), a pair and a random test row, then
step or play through it. At each decision point the text is revealed up to
what the model has read, and the card shows P(switch at the next word) against
the tuned threshold, the predicted duration (Short 1-6 vs Large 7+ by default; a
3-class head can be switched to its own three classes), and the truth.

Two prediction sources (``src/replay.py``):

* ``saved`` (default): ``data/preds/<tag>[.npz]`` from ``--save-preds`` (test split), or
  ``data/preds/lince_<tag>.npz`` from ``eval_lince.py``. Duration was saved only at
  true switches with a known class.
* ``live``: ``models/anticipation_<tag>.pt`` run on CPU, with duration at
  every decision point. Enabled per model when the checkpoint exists (the
  Kaggle notebook keeps it with ``SAVE_CHECKPOINT = True``).

Binds to 127.0.0.1 only: SwitchLingua is gated, so the text must not leave
this machine. Read-only: nothing is written. Standard library apart from
numpy / pyarrow (and torch for live mode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.evaluation import PAIR_KEYS  # noqa: E402
from src.replay import LivePredictor, SavedPreds, discover_runs, random_row, replay_row  # noqa: E402

STATE: dict = {"runs": {}, "saved": {}, "live": {}, "rng": np.random.default_rng(),
               "dataset": "switchlingua"}
LABELS = {"xlmr": "XLM-R", "mbert": "mBERT", "roberta": "RoBERTa",
          "xlmr_bin": "XLM-R (binary)", "mbert_bin": "mBERT (binary)"}
LINCE_PAIRS = ["Spanish->English", "Hindi->English"]


def saved_for(name):
    if name not in STATE["saved"]:
        r = STATE["runs"][name]
        STATE["saved"][name] = SavedPreds(r.preds_path, r.pred_splits)
    return STATE["saved"][name]


def live_for(name):
    if name not in STATE["live"]:
        print(f"loading checkpoint {STATE['runs'][name].ckpt} ...", flush=True)
        STATE["live"][name] = LivePredictor(STATE["runs"][name])
    return STATE["live"][name]


def meta() -> dict:
    return dict(runs=[dict(name=n, label=LABELS.get(n, n), tag=r.tag, modes=r.modes(),
                           tau=round(r.tau, 4), unit=os.path.basename(r.path), n_dur=r.n_dur)
                      for n, r in STATE["runs"].items()],
                dataset=STATE["dataset"],
                pairs=LINCE_PAIRS if STATE["dataset"] == "lince" else PAIR_KEYS)


def pick_row(names, mode, pair):
    if mode == "saved":
        ids = None
        for n in names:
            got = saved_for(n).row_ids(pair or None)
            ids = got if ids is None else np.intersect1d(ids, got)
        if ids is None or not len(ids):
            raise KeyError(f"no saved rows shared by these models for {pair or 'any pair'}")
        return int(STATE["rng"].choice(ids))
    return random_row(STATE["rng"], None, STATE["runs"][names[0]], pair or None)


def api_row(q) -> dict:
    run = q.get("run", "xlmr")
    mode = q.get("mode", "saved")
    if run in ("both", "all"):
        names = [n for n, r in STATE["runs"].items() if mode in r.modes()]
        if not names:
            raise KeyError(f"no model has {mode} predictions")
    else:
        names = [run]
    for n in names:
        if n not in STATE["runs"]:
            raise KeyError(f"unknown model {n!r}")
        if mode not in STATE["runs"][n].modes():
            raise KeyError(f"{LABELS.get(n, n)} has no {mode} predictions"
                           + (" (no checkpoint in models/)" if mode == "live" else ""))
    rid = q.get("row_id")
    rid = int(rid) if rid else pick_row(names, mode, q.get("pair"))
    panels = []
    for n in names:
        r = STATE["runs"][n]
        panels.append(replay_row(r, rid, mode,
                                 saved=saved_for(n) if mode == "saved" else None,
                                 live=live_for(n) if mode == "live" else None)
                      | dict(label=LABELS.get(n, n)))
    return dict(row_id=rid, panels=panels)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html")
            if u.path == "/api/meta":
                return self._send(200, json.dumps(meta()))
            if u.path == "/api/row":
                return self._send(200, json.dumps(api_row(q)))
            return self._send(404, json.dumps({"error": "not found"}))
        except KeyError as e:
            return self._send(400, json.dumps({"error": str(e).strip("'\"")}))
        except Exception as e:                        # show it in the UI, keep serving
            traceback.print_exc()
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switch Replay</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc;
  --l1:#2f6fb0; --l1bg:#e3eef9; --l2:#c2611f; --l2bg:#fbeadd; --none:#8a8a84;
  --tp:#2e8b57; --fp:#d4a017; --fn:#c0392b; --tn:#c9c8c1; --accent:#1d1d1b;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#151514; --panel:#1f1f1d; --ink:#ecebe6; --muted:#9a9993; --line:#33332f;
  --l1:#7fb2e8; --l1bg:#1d3045; --l2:#f0a46b; --l2bg:#43291a; --none:#77766f;
  --tp:#5cc28a; --fp:#e9c24f; --fn:#ef6f5f; --tn:#4a4a45; --accent:#ecebe6;}}
*{box-sizing:border-box}
body{margin:0;overflow-x:hidden;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px 6px;display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:19px;margin:0;font-weight:650;letter-spacing:-.01em}
header .sub{color:var(--muted);font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 24px;border-bottom:1px solid var(--line)}
.bar label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:2px}
select,input,button{font:inherit;font-size:14px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px}
input[type=number]{width:100px}
button{cursor:pointer}
button.primary{background:var(--accent);color:var(--bg);border-color:var(--accent);font-weight:600}
button:disabled{opacity:.4;cursor:default}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;border-radius:0}
.seg button.on{background:var(--accent);color:var(--bg)}
.transport{display:flex;gap:8px;align-items:center;padding:12px 24px;flex-wrap:wrap}
.transport input[type=range]{flex:1;min-width:120px;max-width:100%}
.meta{color:var(--muted);font-size:13px;padding:0 24px}
#panels{display:grid;gap:16px;padding:12px 24px 28px;grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr))}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.ph{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.ph h2{font-size:16px;margin:0}
.ph .s{font-size:12px;color:var(--muted)}
.text{margin:10px 0;padding:10px 12px;border:1px solid var(--line);border-radius:8px;max-height:230px;overflow:auto;font-size:15.5px;line-height:1.9;white-space:pre-wrap;word-break:break-word}
.pc{border-radius:3px;padding:1px 0}
.pc.l0{color:var(--l1);background:var(--l1bg)}
.pc.l1{color:var(--l2);background:var(--l2bg)}
.pc.l2{color:var(--none)}
.pc.fut{opacity:.18;background:none}
.pc.cur{outline:2px solid var(--ink);outline-offset:1px}
.legend{display:flex;gap:12px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.card{display:grid;gap:8px;margin-top:6px}
.row2{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.k{font-size:12px;color:var(--muted);min-width:92px}
.pbar{position:relative;flex:1;min-width:120px;height:14px;background:var(--line);border-radius:7px;overflow:hidden}
.pbar .f{position:absolute;inset:0 auto 0 0;background:var(--l2);border-radius:7px}
.pbar .t{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}
.num{font-variant-numeric:tabular-nums;font-size:13px;min-width:46px;text-align:right}
.badge{font-weight:700;font-size:12px;padding:2px 8px;border-radius:10px;color:#fff}
.b-TP{background:var(--tp)}.b-FP{background:var(--fp);color:#1d1d1b}.b-FN{background:var(--fn)}.b-TN{background:var(--tn);color:var(--ink)}
.verdict{font-size:14px}
.dur{display:grid;grid-template-columns:118px 1fr 46px;gap:4px 8px;align-items:center;font-size:13px}
.dur .pbar{height:10px}
.dur .truth{font-weight:700}
.dur .truth::after{content:" ◀ true";font-weight:400;color:var(--muted);font-size:11px}
.note{font-size:12px;color:var(--muted)}
svg.tl{width:100%;height:58px;display:block;margin-top:10px;cursor:pointer}
.sum{font-size:13px;display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;color:var(--muted)}
.sum b{color:var(--ink)}
.err{margin:12px 24px;padding:10px 12px;border-radius:8px;background:var(--l2bg);color:var(--ink);display:none}
.empty{padding:40px 24px;color:var(--muted)}
@media (max-width:520px){header,.bar,.transport,#panels,.meta{padding-left:16px;padding-right:16px}#panels{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>Switch Replay</h1><span class="sub">Anticipatory code-switch predictions, replayed one decision point at a time. Local only.</span></header>
<div class="bar">
  <label>Model<span class="seg" id="model"></span></label>
  <label>Source<span class="seg" id="mode"><button data-v="saved" class="on">saved</button><button data-v="live">live</button></span></label>
  <label>Duration<span class="seg" id="durview"><button data-v="bin" class="on">Short / Large</button><button data-v="native">3 classes</button></span></label>
  <label>Pair<select id="pair"><option value="">any pair</option></select></label>
  <button class="primary" id="rand">🎲 Random row</button>
  <label>Row id<input type="number" id="rid" placeholder="row_id"></label>
  <button id="load">Load</button>
</div>
<div class="err" id="err"></div>
<div class="transport">
  <button id="prev" title="previous decision (←)">◀</button>
  <button id="play" title="play / pause (space)">▶ Play</button>
  <button id="next" title="next decision (→)">▶</button>
  <input type="range" id="scrub" min="0" max="0" value="0">
  <span class="num" id="pos">0/0</span>
  <label>Speed<select id="speed"><option value="1200">slow</option><option value="600" selected>normal</option><option value="250">fast</option></select></label>
</div>
<div class="meta" id="meta"></div>
<div id="panels"><div class="empty">Pick a model and press <b>Random row</b>.</div></div>
<script>
const $=s=>document.querySelector(s);
let META=null, DATA=null, steps=[], cur=0, timer=null, model="xlmr", mode="saved", durview="bin";
const DUR_RANGE={small:"1–2",medium:"3–6",large:"7+",short:"1–6"};

function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function err(m){const e=$("#err");e.textContent=m||"";e.style.display=m?"block":"none"}

async function init(){
  META=await (await fetch("/api/meta")).json();
  const seg=$("#model");
  const opts=META.runs.map(r=>[r.name,r.label]);
  if(META.runs.length>1) opts.push(["all",META.runs.length==2?"Both":"All"]);
  seg.innerHTML=opts.map(([v,l])=>`<button data-v="${v}">${l}</button>`).join("");
  model=META.runs[0]?.name||"xlmr";
  seg.onclick=e=>{if(e.target.dataset.v){model=e.target.dataset.v;syncSeg();if(DATA)load(DATA.row_id)}};
  $("#mode").onclick=e=>{const b=e.target;if(b.dataset.v&&!b.disabled){mode=b.dataset.v;syncSeg();if(DATA)load(DATA.row_id)}};
  $("#durview").onclick=e=>{const b=e.target;if(b.dataset.v&&!b.disabled){durview=b.dataset.v;syncSeg();if(DATA)render()}};
  META.pairs.forEach(p=>$("#pair").insertAdjacentHTML("beforeend",`<option>${p}</option>`));
  if(META.dataset==="lince") document.querySelector("header .sub").textContent="LinCE tweets (human gold language tags) — out-of-domain replay. Local only.";
  const u=new URLSearchParams(location.search);       // deep link: ?run=both&mode=saved&row=123&step=10
  if(u.get("run")) model=u.get("run")==="both"?"all":u.get("run");
  if(u.get("dur")) durview=u.get("dur");
  if(u.get("mode")) mode=u.get("mode");
  if(u.get("pair")) $("#pair").value=u.get("pair");
  syncSeg();
  if(u.get("row")){await load(u.get("row")); if(u.get("step")){cur=Math.min(steps.length-1,+u.get("step"));render()}}
}
function selRuns(){return model==="all"?META.runs:META.runs.filter(r=>r.name===model)}
function syncSeg(){
  document.querySelectorAll("#model button").forEach(b=>b.classList.toggle("on",b.dataset.v===model));
  const live=selRuns().every(r=>r.modes.includes("live"));
  const lb=$('#mode [data-v="live"]'); lb.disabled=!live;
  lb.title=live?"run the checkpoint on CPU":"no checkpoint in models/ — re-run Kaggle with SAVE_CHECKPOINT = True and download models/anticipation_<tag>.pt";
  if(!live&&mode==="live") mode="saved";
  document.querySelectorAll("#mode button").forEach(b=>b.classList.toggle("on",b.dataset.v===mode));
  const any3=selRuns().some(r=>r.n_dur===3), nb=$('#durview [data-v="native"]');
  nb.disabled=!any3; nb.title=any3?"the head's own Small / Medium / Large":"binary heads only predict Short / Large";
  if(!any3&&durview==="native") durview="bin";
  document.querySelectorAll("#durview button").forEach(b=>b.classList.toggle("on",b.dataset.v===durview));
}

async function load(rowId){
  stop(); err("");
  const q=new URLSearchParams({run:model,mode});
  if(rowId!=null&&rowId!=="") q.set("row_id",rowId); else if($("#pair").value) q.set("pair",$("#pair").value);
  $("#panels").innerHTML=`<div class="empty">${mode==="live"?"Running the model…":"Loading…"}</div>`;
  const r=await fetch("/api/row?"+q); const j=await r.json();
  if(!r.ok){err(j.error);$("#panels").innerHTML="";return}
  DATA=j; $("#rid").value=j.row_id; prepare(); render();
  history.replaceState(null,"","?"+new URLSearchParams({run:model,mode,row:j.row_id,dur:durview}));
}

// A shared clock across panels: progress is the number of non-space characters read,
// so two tokenizers' decision points line up on the same text.
function prepare(){
  const all=new Set();
  for(const p of DATA.panels){
    let c=0; p.off=p.pieces.map(s=>(c+=s.replace(/\s/g,"").length));
    p.dOff=p.decisions.map(d=>p.off[d.pos]);
    p.dOff.forEach(o=>all.add(o));
  }
  steps=[...all].sort((a,b)=>a-b); cur=0;
  $("#scrub").max=Math.max(0,steps.length-1);
  const p0=DATA.panels[0];
  $("#meta").innerHTML=`Row <b>${DATA.row_id}</b> · ${esc(p0.pair)} · ${p0.dataset==="lince"?"LinCE (gold labels)":"SwitchLingua test split"} · source <b>${p0.mode}</b>`+
    (p0.mode==="saved"?" · duration shown at true switches only":" · duration at every decision point");
}
function idxFor(p){let k=-1;for(let i=0;i<p.dOff.length;i++){if(p.dOff[i]<=steps[cur])k=i;else break}return k}

function render(){
  $("#scrub").value=cur; $("#pos").textContent=`${steps.length?cur+1:0}/${steps.length}`;
  $("#panels").innerHTML=DATA.panels.map((p,i)=>panel(p,i)).join("");
  document.querySelectorAll("svg.tl").forEach(svg=>svg.onclick=e=>{
    const p=DATA.panels[+svg.dataset.i], r=svg.getBoundingClientRect();
    const k=Math.max(0,Math.min(p.decisions.length-1,Math.floor((e.clientX-r.left)/r.width*p.decisions.length)));
    cur=steps.indexOf(p.dOff[k]); render();
  });
  const c=document.querySelector(".pc.cur"); if(c) c.scrollIntoView({block:"nearest"});
}

function panel(p,pi){
  const k=idxFor(p), d=k>=0?p.decisions[k]:null, upto=d?d.pos:-1;
  const L=[p.first_language,p.second_language];
  const txt=p.pieces.map((s,i)=>`<span class="pc l${p.piece_lang[i]}${i>upto?" fut":""}${i===upto?" cur":""}">${esc(s)}</span>`).join("");
  const s=p.summary;
  return `<div class="panel">
   <div class="ph"><h2>${esc(p.label)}</h2><span class="s">${esc(p.tag)} · τ = ${p.tau}</span></div>
   <div class="legend"><span><i class="sw" style="background:var(--l1)"></i>${esc(L[0])} (L1)</span><span><i class="sw" style="background:var(--l2)"></i>${esc(L[1])} (L2)</span><span><i class="sw" style="background:var(--none)"></i>punctuation / none</span><span>faded = not yet read</span></div>
   <div class="text">${txt}</div>
   ${d?card(p,d,L):`<div class="note">The model hasn't reached its first decision point yet.</div>`}
   ${timeline(p,k,pi)}
   <div class="sum">Row: <span><b>${s.TP}</b> TP</span><span><b>${s.FP}</b> FP</span><span><b>${s.FN}</b> FN</span><span><b>${s.TN}</b> TN</span><span>switch F1 <b>${s.f1==null?"—":s.f1.toFixed(2)}</b></span><span>${s.n} decisions</span></div>
  </div>`;
}

function card(p,d,L){
  const pct=(d.p*100).toFixed(1), from=d.next_lang==null?null:(d.y?1-d.next_lang:d.next_lang);
  const toName=from==null?"the other language":L[1-from];
  const verdict=d.fired?`predicts a <b>switch</b> to ${esc(toName)} at the next word`:`predicts <b>no switch</b>: next word stays ${from==null?"":"in "+esc(L[from])}`;
  let truth;
  if(d.y){
    const len=d.y_len+(d.censored?"+ (cut off by end of text)":"");
    const bin=durview==="bin"||p.n_dur===2;
    const cls=bin?(d.y_bin!=null?` — ${["short","large"][d.y_bin]}`:""):(d.y_dur!=null?` — ${p.dur_labels[d.y_dur]}`:"");
    truth=`switch to ${esc(L[d.next_lang])}, run of <b>${len}</b> pieces${cls}`;
  } else truth="no switch";
  let dur="";
  const bin=durview==="bin"||p.n_dur===2;
  const probs=bin?d.bin_probs:d.dur_probs, pred=bin?d.bin_pred:d.dur_pred, yt=bin?d.y_bin:d.y_dur;
  const names=bin?["short","large"]:p.dur_labels;
  if(probs){
    const rule=bin&&p.n_dur===3?`Large if P(large) > ${p.large_cut}`:"";
    const why=d.fired||mode==="live"?"":"(model did not fire here — shown for reference)";
    dur=`<div class="row2"><span class="k">Duration</span><span class="note">${why} ${rule}</span></div><div class="dur">`+
      names.map((n,i)=>`<span class="${yt===i?"truth":""}">${n}${pred===i?" ●":""} <span class="note">${DUR_RANGE[n]||""}</span></span><div class="pbar"><div class="f" style="width:${probs[i]*100}%;${pred===i?"":"opacity:.45"}"></div></div><span class="num">${(probs[i]*100).toFixed(0)}%</span>`).join("")+`</div>`;
  } else if(d.fired){
    dur=`<div class="note">Duration: not saved at this position (saved predictions keep duration only at true switches). Use live mode for every point.</div>`;
  }
  return `<div class="card">
   <div class="row2"><span class="k">P(switch next)</span><div class="pbar"><div class="f" style="width:${pct}%"></div><div class="t" style="left:${p.tau*100}%" title="threshold τ"></div></div><span class="num">${pct}%</span><span class="badge b-${d.outcome}">${d.outcome}</span></div>
   <div class="verdict">Model ${verdict}.</div>
   <div class="verdict"><span class="k">Truth</span> ${truth}.</div>
   ${dur}
  </div>`;
}

function timeline(p,k,pi){
  const n=p.decisions.length; if(!n) return "";
  const W=1000,H=58,w=W/n;
  const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v);
  const col={TP:css("--tp"),FP:css("--fp"),FN:css("--fn"),TN:css("--tn")};
  let g="";
  p.decisions.forEach((d,i)=>{const h=Math.max(2,d.p*(H-8));g+=`<rect x="${i*w+w*.12}" y="${H-h}" width="${Math.max(1,w*.76)}" height="${h}" fill="${col[d.outcome]}"${i>k?' opacity=".35"':""}><title>${d.outcome} p=${d.p}</title></rect>`});
  const ty=H-p.tau*(H-8);
  g+=`<line x1="0" x2="${W}" y1="${ty}" y2="${ty}" stroke="${css("--ink")}" stroke-dasharray="4 4" stroke-width="1"/>`;
  if(k>=0) g+=`<rect x="${k*w}" y="0" width="${Math.max(2,w)}" height="${H}" fill="none" stroke="${css("--ink")}" stroke-width="2"/>`;
  return `<svg class="tl" data-i="${pi}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="P(switch) at each decision point">${g}</svg>
   <div class="legend"><span>bar height = P(switch), dashed = τ</span><span><i class="sw" style="background:var(--tp)"></i>TP</span><span><i class="sw" style="background:var(--fp)"></i>FP</span><span><i class="sw" style="background:var(--fn)"></i>FN</span><span><i class="sw" style="background:var(--tn)"></i>TN</span></div>`;
}

function step(n){if(!DATA)return;cur=Math.max(0,Math.min(steps.length-1,cur+n));render()}
function stop(){clearInterval(timer);timer=null;$("#play").textContent="▶ Play"}
function play(){
  if(!DATA)return; if(timer){stop();return}
  if(cur>=steps.length-1) cur=0;
  $("#play").textContent="❚❚ Pause";
  timer=setInterval(()=>{if(cur>=steps.length-1){stop();return}step(1)},+$("#speed").value);
}
$("#rand").onclick=()=>load(null);
$("#load").onclick=()=>load($("#rid").value);
$("#rid").onkeydown=e=>{if(e.key==="Enter")load($("#rid").value)};
$("#prev").onclick=()=>step(-1); $("#next").onclick=()=>step(1); $("#play").onclick=play;
$("#scrub").oninput=e=>{cur=+e.target.value;render()};
$("#speed").onchange=()=>{if(timer){stop();play()}};
document.onkeydown=e=>{if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  if(e.key==="ArrowRight")step(1); else if(e.key==="ArrowLeft")step(-1); else if(e.key===" "){e.preventDefault();play()}};
init();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=TAG",
                    help="add or override a model, e.g. roberta=roberta_gap (default: "
                         "xlmr=xlmr_prev_gap_36k_rsv2, mbert=mbert_prev_gap_36k_rsv2)")
    ap.add_argument("--seed", type=int, default=None, help="seed the random row picker")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    ap.add_argument("--dataset", choices=["switchlingua", "lince"], default="switchlingua",
                    help="lince: replay LinCE tweets (human gold tags) instead of SwitchLingua test rows")
    args = ap.parse_args()
    extra = dict(s.split("=", 1) for s in args.run)
    STATE["dataset"] = args.dataset
    STATE["runs"] = discover_runs(extra, dataset=args.dataset)
    STATE["rng"] = np.random.default_rng(args.seed)
    if not STATE["runs"]:
        raise SystemExit("no runs found: need data/results/anticipation_<tag>.json and "
                         + ("data/preds/lince_<tag>.npz (eval_lince.py --save-preds)"
                            if args.dataset == "lince" else "data/preds/<tag>[.npz]")
                         + " or models/anticipation_<tag>.pt")
    for n, r in STATE["runs"].items():
        print(f"  {LABELS.get(n, n):8} {r.tag:28} tau={r.tau:.3f}  sources: {', '.join(r.modes())}")
    url = f"http://localhost:{args.port}"
    print(f"\n  ->  {url}\n\nCtrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
