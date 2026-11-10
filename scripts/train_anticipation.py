#!/usr/bin/env python
"""Train an anticipation model and score it with the shared evaluation report.

    # the from-scratch causal transformer (words, or XLM-R pieces)
    python scripts/train_anticipation.py --build-vocab --path data/tagged_shuffled.parquet --steps 6000
    python scripts/train_anticipation.py --unit subword --build-vocab --steps 6000
    # XLM-R with a causal mask (needs a GPU: see kaggle/train_xlmr.ipynb)
    python scripts/train_anticipation.py --model xlmr --unit subword --steps 12000 --batch-size 16

Protocol (identical for every model): train on the train split; tune the
switch threshold on val; score test once with ``src.evaluation.full_report``
(switch P/R/F1 per pair, macro-F1, per direction; 3-class duration at true
switch points). ``check_causality`` must pass before training starts.
Everything is streamed; the process never holds more than a shuffle buffer.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import N_DUR  # noqa: E402
from src.causal_model import (check_causality, check_context_free, duration_probs,  # noqa: E402
                              ordinal_targets)
from src.evaluation import format_duration, format_report, full_report  # noqa: E402
from src.metrics import apply_class_offsets, duration_scores, prf, tune_class_offsets  # noqa: E402
from src.models import build_model, pair_ids  # noqa: E402
from src.streaming import (DEFAULT_PATHS, VOCAB_PATHS, build_subword_vocab,  # noqa: E402
                           build_vocab, make_loader)
from src.training import (count_duration_classes, duration_class_weights,  # noqa: E402
                          save_predictions, set_seed)

UNIT_DEFAULTS = {"word": dict(window=256, stride=192, vocab_size=50_000,
                              out="data/anticipation_word.json",
                              ckpt="models/anticipation_word.pt"),
                 "subword": dict(window=384, stride=288, vocab_size=100_000,
                                 out="data/anticipation_model_subword.json",
                                 ckpt="models/anticipation_subword.pt")}
MODEL_DEFAULTS = {"scratch": dict(lr=3e-4),
                  "xlmr": dict(lr=2e-5, out="data/anticipation_xlmr.json",
                               ckpt="models/anticipation_xlmr.pt"),
                  "roberta": dict(lr=2e-5, path="data/subword_roberta.parquet",
                                  out="data/anticipation_roberta.json",
                                  ckpt="models/anticipation_roberta.pt"),
                  "mbert": dict(lr=2e-5, path="data/subword_mbert.parquet",
                                out="data/anticipation_mbert.json",
                                ckpt="models/anticipation_mbert.pt")}
PRETRAINED = ("xlmr", "roberta", "mbert")    # each reads its own tokenizer's pieces
PAIRS = ["Cantonese", "Arabic", "Philippines", "German", "Chinese", "Vietnamese",
         "Malay", "Japanese", "Hindi", "Korean", "Spanish", "French", "Russian",
         "Italian"]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def resolve_defaults(args):
    """Fill model- and unit-dependent defaults; no run overwrites another model's files."""
    model = getattr(args, "model", "scratch")
    if model in PRETRAINED and args.unit != "subword":
        raise SystemExit(f"--model {model} reads sub-word pieces: use --unit subword")
    context = getattr(args, "context", None)
    explicit = {k for k in ("out", "ckpt") if getattr(args, k, None) is not None}
    pretrained_only = [f for f, on in (("--pair-emb", getattr(args, "pair_emb", False)),
                                       ("--lora-r", getattr(args, "lora_r", 0))) if on]
    if pretrained_only and model not in PRETRAINED:
        raise SystemExit(f"{', '.join(pretrained_only)} apply to the pretrained encoders only")
    if getattr(args, "lora_r", 0) and getattr(args, "lr", None) is None:
        args.lr = 2e-4                           # adapters start at zero and need a larger step
    for k, v in {**UNIT_DEFAULTS[args.unit], **MODEL_DEFAULTS[model]}.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.path is None:
        args.path = DEFAULT_PATHS[args.unit]
    if args.unit == "subword" and args.names != "script_words":
        # data/subword.parquet was built with the default names policy baked in.
        raise SystemExit("--names only applies to --unit word: the sub-word labels were "
                         "built with script_words (use kaggle/load_labels.ipynb for others)")
    subword_only = [f for f, on in (("--turn-sep", getattr(args, "turn_sep", False)),
                                    ("--prev-lang", getattr(args, "prev_lang", False)),
                                    ("--predict-at word_end",
                                     getattr(args, "predict_at", "piece") == "word_end"),
                                    ("--decide-at gap", getattr(args, "decide_at", "word") == "gap")) if on]
    if subword_only and args.unit != "subword":
        raise SystemExit(f"{', '.join(subword_only)} need the sub-word labels: use --unit subword")
    if context is not None:
        if model not in PRETRAINED:
            raise SystemExit("--context applies to the pretrained encoders only")
        # Each of these gives the current-token model information beyond piece t:
        # a separator or the previous word's language is context, and scoring only
        # word ends tells it that the next piece starts a new word.
        leaks = [f for f, on in (("--turn-sep", getattr(args, "turn_sep", False)),
                                 ("--prev-lang", getattr(args, "prev_lang", False)),
                                 ("--predict-at word_end",
                                  getattr(args, "predict_at", "piece") == "word_end")) if on]
        if leaks:
            raise SystemExit(f"--context 1 reads piece t only: {', '.join(leaks)} would add context")
        for k in ("out", "ckpt"):                # no run overwrites the full-prefix model
            if k not in explicit:
                root, ext = os.path.splitext(getattr(args, k))
                setattr(args, k, f"{root}_ctx{context}{ext}")
    if args.window > 512 and model in PRETRAINED:
        raise SystemExit("pretrained encoders read at most 512 positions: use --window 512 or less")
    return args


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def autocast(dev):
    """fp16 on CUDA (a T4 has no bf16); full precision elsewhere."""
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def scored_positions(b):
    """``(row_id, position in the row)`` of every scored position in a batch, in order."""
    m = b["sw_mask"]
    row = b["row_id"].unsqueeze(1).expand_as(m)[m]
    pos = (b["start"].unsqueeze(1) + torch.arange(m.shape[1])).expand_as(m)[m]
    return row.numpy(), pos.numpy()


def curve_point(model, loader, dev, n_batches: int, step: int) -> dict:
    """Validation snapshot during training (``--eval-every``), on a fixed val slice.

    The slice is the first ``n_batches`` of the val split, in file order, so
    every snapshot scores the same positions. The F1 threshold is tuned on that
    same slice, which is optimistic but equally so at every step: use the curve's
    shape, not its level.

    Starting a DataLoader iterator draws one number from torch's global RNG,
    which would shift every later dropout mask. ``fork_rng`` restores the RNG
    state afterwards, so a seeded run gives the same model with or without
    ``--eval-every``.
    """
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        res = evaluate(model, loader, dev, n_batches)
    model.train()
    logit, y = res["logit"].astype(np.float64), res["y"].astype(np.float64)
    loss = float(np.mean(np.logaddexp(0, logit) - y * logit))     # BCE from logits
    tau, f1 = tune_threshold(1 / (1 + np.exp(-logit)), res["y"])
    dur = duration_scores(res["dtrue"], res["dpred"]) if len(res["dtrue"]) else {"macro_f1": 0.0}
    # Duration loss: unweighted 3-class cross-entropy at the dur_mask positions, the same
    # quantity as L_dur in training with --dur-class-weights none (for an ordinal head,
    # the NLL of its derived class probabilities). None if the slice has no such position.
    nd = len(res["dtrue"])
    p_true = res["dprob"][np.arange(nd), res["dtrue"].astype(int)] if nd else None
    dur_loss = float(-np.mean(np.log(np.clip(p_true, 1e-12, None)))) if nd else None
    return dict(step=step, val_switch_loss=loss, val_f1=f1, tau=tau,
                val_dur_macro_f1=dur["macro_f1"], val_dur_loss=dur_loss,
                n=int(len(y)), n_dur=int(nd))


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=0):
    """Logits and labels at every scored position, with pair and direction."""
    model.eval()
    out = {k: [] for k in ("logit", "y", "pair", "lang", "dtrue", "dpred", "dprob", "dpair",
                           "dlang", "dlogit", "drow", "shared", "row", "pos")}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with autocast(dev):
            sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev), pair_ids(model, b).to(dev))
        m = b["sw_mask"]
        out["logit"].append(sw.float().cpu()[m].numpy())
        out["y"].append(b["y_sw"][m].numpy())
        out["pair"].append(b["report_pair"].unsqueeze(1).expand_as(m)[m].numpy())
        out["lang"].append(b["cur_lang"][m].numpy())        # direction: current language
        dm = b["dur_mask"]
        if dm.any():
            out["dtrue"].append(b["y_dur"][dm].numpy())
            probs = duration_probs(dur.float().cpu()[dm], getattr(model, "dur_ordinal", False))
            out["dpred"].append(probs.argmax(-1).numpy())
            out["dprob"].append(probs.numpy())
            # aligned with dtrue: dur_mask is a subset of sw_mask
            out["dpair"].append(b["report_pair"].unsqueeze(1).expand_as(dm)[dm].numpy())
            out["dlang"].append(b["cur_lang"][dm].numpy())
            out["dlogit"].append(sw.float().cpu()[dm].numpy())
            out["drow"].append(b["row_id"].unsqueeze(1).expand_as(dm)[dm].numpy())
        out["shared"].append(b["shared_mask"][m].numpy())
        row, pos = scored_positions(b)                      # trace errors back to the text
        out["row"].append(row)
        out["pos"].append(pos)
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


def make_scheduler(opt, lrs, steps):
    """One-cycle schedule; the warm-up is at least 2 steps (OneCycleLR divides by it)."""
    if steps < 4:
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=steps,
                                               pct_start=min(0.5, max(0.1, 2.5 / steps)))


def optimizer_groups(model, args):
    """``(param groups, max lrs)``. A pretrained body learns slowly and the new heads fast;
    ``--freeze-backbone`` trains the heads only (a linear probe of the pretrained features)."""
    if args.model not in PRETRAINED:
        if args.freeze_backbone:
            raise SystemExit("--freeze-backbone needs a pretrained --model")
        return [dict(params=list(model.parameters()), lr=args.lr)], [args.lr]
    if getattr(args, "lora_r", 0) and args.freeze_backbone:
        raise SystemExit("--lora-r and --freeze-backbone are exclusive: LoRA already freezes the body")
    heads = [p for n, p in model.named_parameters() if n.startswith("head_")]
    # requires_grad: with --lora-r the pretrained weights are frozen and only the
    # adapters (and the new language / pair embeddings) train at --lr.
    body = [p for n, p in model.named_parameters() if not n.startswith("head_") and p.requires_grad]
    if args.freeze_backbone:
        for p in body:
            p.requires_grad_(False)
        return [dict(params=heads, lr=args.head_lr)], [args.head_lr]
    return [dict(params=body, lr=args.lr), dict(params=heads, lr=args.head_lr)], [args.lr, args.head_lr]


def tune_threshold(prob, truth):
    """Pick the F1-maximising threshold on a held-out split."""
    best = (0.0, 0.5)
    for t in np.quantile(prob, np.linspace(0.50, 0.999, 60)):
        best = max(best, (prf(truth, prob >= t)["f1"], float(t)))
    return best[1], best[0]


def score(res, tau):
    """The full report, plus ``shared``: the same report on the positions every
    model shares (the last piece of each non-CJK word), comparable across tokenizers."""
    pred = (1 / (1 + np.exp(-res["logit"])) >= tau).astype(int)
    # Duration side arrays are aligned with dtrue; older prediction files lack them.
    def aligned(k):
        return res[k] if k in res and len(res[k]) == len(res["dtrue"]) else None

    dlogit = aligned("dlogit")
    fired = None if dlogit is None else 1 / (1 + np.exp(-np.asarray(dlogit))) >= tau
    rep = full_report(res["y"], pred, res["pair"], res["lang"], res["dtrue"], res["dpred"],
                      dur_pairs=aligned("dpair"), dur_langs=aligned("dlang"), dur_fired=fired)
    sh = np.asarray(res.get("shared", np.zeros(len(pred), bool)), dtype=bool)
    if sh.any():
        rep["shared"] = full_report(res["y"][sh], pred[sh], res["pair"][sh], res["lang"][sh])
    return rep


DURATION_KEYS = ("duration", "duration_end_to_end", "duration_per_pair", "duration_per_direction")


def tuned_duration(res, tau, offsets, val_macro_f1) -> dict:
    """The duration report again, after ``offsets`` (fit on val) move the decision."""
    rep = score(dict(res, dpred=apply_class_offsets(res["dprob"], offsets)), tau)
    return dict(offsets=np.asarray(offsets).tolist(), val_macro_f1=float(val_macro_f1),
                report={k: rep[k] for k in DURATION_KEYS if k in rep})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scratch", *PRETRAINED], default="scratch")
    ap.add_argument("--backbone", default=None, help="override the Hugging Face checkpoint")
    ap.add_argument("--path", default=None)
    ap.add_argument("--build-vocab", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 scratch, 2e-5 pretrained")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="pretrained: learning rate of the heads")
    ap.add_argument("--dur-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batches", type=int, default=0, help="0 = score the whole split")
    ap.add_argument("--train-langs", default="",
                    help="comma-separated first_languages to TRAIN on (default: all). "
                         "Everything else becomes the zero-shot set.")
    ap.add_argument("--no-pair", action="store_true",
                    help="drop the language-pair embedding (scratch model)")
    ap.add_argument("--unit", choices=["word", "subword"], default="word")
    ap.add_argument("--no-lang", action="store_true",
                    help="no language input (implied by --unit subword)")
    ap.add_argument("--out", default=None, help="default depends on --model and --unit")
    ap.add_argument("--ckpt", default=None, help="default depends on --model and --unit")
    ap.add_argument("--seed", type=int, default=None, help="seed initialisation and data order")
    ap.add_argument("--save-preds", default=None,
                    help="save val/test predictions (.npz) for offline threshold tuning")
    ap.add_argument("--dur-class-weights", choices=["none", "balanced"], default="none",
                    help="balanced: inverse-frequency weights from the train split")
    ap.add_argument("--turn-sep", action="store_true", help="separator piece between turns (pieces)")
    ap.add_argument("--prev-lang", action="store_true",
                    help="input: the language of the last finished word (pieces)")
    ap.add_argument("--predict-at", choices=["piece", "word_end"], default="piece",
                    help="word_end: train and score only at the last piece of each word")
    ap.add_argument("--decide-at", choices=["word", "gap"], default="word",
                    help="gap: put each label on the punctuation after its word, so the model has "
                         "read it (pieces; see anticipation.gap_landing)")
    ap.add_argument("--dur-head", choices=["softmax", "ordinal"], default="softmax",
                    help="ordinal: cumulative cuts P(>Small), P(>Medium) instead of a 3-way softmax")
    ap.add_argument("--context", type=int, choices=[1], default=None,
                    help="pretrained: 1 = read the current piece only (no prefix); default the whole prefix")
    ap.add_argument("--reshuffle-epochs", action="store_true",
                    help="a new data order every epoch (off: each epoch repeats the first's order)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="score a fixed val slice every N steps (learning curve in the json)")
    ap.add_argument("--eval-every-batches", type=int, default=200,
                    help="size of that val slice, in batches")
    ap.add_argument("--pair-emb", action="store_true",
                    help="pretrained: add a learned embedding of the row's language pair (15 pairs)")
    ap.add_argument("--lora-r", type=int, default=0,
                    help="pretrained: freeze the encoder, train rank-r LoRA adapters (lr default 2e-4)")
    ap.add_argument("--lora-alpha", type=float, default=None, help="LoRA scale numerator (default 2r)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="pretrained: train the two heads only, the backbone stays as pretrained "
                         "(with --steps 0 nothing is trained: random heads, val-tuned decisions)")
    ap.add_argument("--names", choices=["script_words", "transparent", "script", "hybrid", "language"],
                    default="script_words",
                    help="how named entities count (see src/segments.py); word units only")
    args = ap.parse_args()
    import src.anticipation as _ant
    _ant.NAMES = args.names
    resolve_defaults(args)
    if args.seed is not None:
        set_seed(args.seed)
    use_lang = args.prev_lang or not (args.no_lang or args.unit == "subword")

    if args.model in PRETRAINED:
        vocab, vocab_size = None, None                # raw piece ids; size from the backbone
    else:
        vocab_path = VOCAB_PATHS[args.unit]
        if args.build_vocab or not os.path.exists(vocab_path):
            t0 = time.time()
            print("building vocabulary from the train split ...", flush=True)
            builder = build_vocab if args.unit == "word" else build_subword_vocab
            vocab = builder(args.path, args.vocab_size)
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            json.dump(vocab, open(vocab_path, "w"))
            print(f"  {len(vocab):,} types in {time.time()-t0:.0f}s -> {vocab_path}")
        vocab = json.load(open(vocab_path))
        vocab_size = len(vocab)


    train_langs = [x.strip() for x in args.train_langs.split(",") if x.strip()]
    holdout = [p for p in PAIRS if p not in train_langs] if train_langs else []
    if holdout:
        print(f"train languages : {', '.join(train_langs)}")
        print(f"zero-shot held out: {', '.join(holdout)}")

    dev = device_of(args.device)
    model = build_model(vars(args), vocab_size, n_pairs=len(PAIRS),
                        use_pair=args.model == "scratch" and not (args.no_pair or bool(holdout)),
                        use_lang=use_lang)
    vocab_size = vocab_size or model.vocab_size
    print(f"model {args.model} | unit {args.unit} | {args.path} | vocab {vocab_size:,} | "
          f"language input {'on' if use_lang else 'off'}")
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device {dev} | {n_param/1e6:.1f}M params, {n_train/1e6:.1f}M trainable")

    chk = check_causality(model, vocab_size, seq_len=32, device="cpu")
    print(f"causality check: max logit drift {chk['max_abs_logit_drift']:.2e} "
          f"over {chk['positions_tested']} prefixes -> "
          f"{'PREFIX-ONLY OK' if chk['causal'] else 'LEAK'}")
    if not chk["causal"]:
        raise SystemExit("aborting: model is not causal")
    if args.context == 1:                              # passes the check above trivially
        cf = check_context_free(model, vocab_size, seq_len=32, device="cpu")
        print(f"context-free check: max logit drift {cf['max_abs_logit_drift']:.2e} when every "
              f"other position changes -> {'CURRENT PIECE ONLY OK' if cf['context_free'] else 'LEAK'}")
        if not cf["context_free"]:
            raise SystemExit("aborting: the current-token model reads other positions")
        chk["context_free"] = cf
    model.to(dev)

    sep_id = None
    if args.turn_sep and args.model in PRETRAINED:     # the backbone's own separator token
        from transformers import AutoTokenizer
        from src.xlmr_model import BACKBONES
        sep_id = AutoTokenizer.from_pretrained(args.backbone or BACKBONES[args.model]).sep_token_id
    blank_ids = None
    if args.decide_at == "gap":                        # the tokenizer's bare space markers
        from src.subword import blank_piece_ids
        from src.xlmr_model import BACKBONES
        blank_ids = blank_piece_ids(
            (args.backbone or BACKBONES[args.model]) if args.model in PRETRAINED else BACKBONES["xlmr"])
        print(f"decide_at=gap: {len(blank_ids)} blank piece ids {blank_ids[:8]}")
    common = dict(window=args.window, stride=args.stride, pair_index=PAIR_INDEX, unit=args.unit,
                  predict_at=args.predict_at, turn_sep=args.turn_sep, sep_id=sep_id,
                  decide_at=args.decide_at, blank_ids=blank_ids,
                  lang_input="prev_word" if args.prev_lang else "current",
                  seed=args.seed if args.seed is not None else 0)
    train_kw = dict(reshuffle=args.reshuffle_epochs)
    print(f"flags: seed={args.seed} predict_at={args.predict_at} decide_at={args.decide_at} turn_sep={args.turn_sep} "
          f"prev_lang={args.prev_lang} dur_class_weights={args.dur_class_weights} "
          f"context={args.context or 'prefix'} dur_head={args.dur_head} "
          f"freeze_backbone={args.freeze_backbone} pair_emb={args.pair_emb} lora_r={args.lora_r} "
          f"reshuffle_epochs={args.reshuffle_epochs}")
    loader = lambda split, langs, workers=0: make_loader(  # noqa: E731
        args.path, vocab, split, args.batch_size, workers, languages=langs, **common)
    train_loader = make_loader(args.path, vocab, "train", args.batch_size, args.workers,
                               languages=train_langs or None, **common, **train_kw)
    val_loader, test_loader = loader("val", train_langs or None), loader("test", train_langs or None)
    curve_loader = val_loader                 # unshuffled: each pass restarts at the same rows

    groups, lrs = optimizer_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = make_scheduler(opt, lrs, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    dur_w = None
    if args.dur_class_weights == "balanced":
        counts = count_duration_classes(args.path, args.unit, train_langs or None, args.predict_at)
        dur_w = duration_class_weights(counts)
        print("duration class counts (train):", counts.tolist(), "-> weights", np.round(dur_w, 3).tolist())
        dur_w = torch.tensor(dur_w, dtype=torch.float32, device=dev)
    ce = nn.CrossEntropyLoss(weight=dur_w, reduction="none")
    ordinal = args.dur_head == "ordinal"

    step, t0, running, curve = 0, time.time(), [], []
    model.train()
    while step < args.steps:
        for b in train_loader:
            if step >= args.steps:
                break
            with autocast(dev):
                sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev),
                                pair_ids(model, b).to(dev))
            sw, dur = sw.float(), dur.float()
            m = b["sw_mask"].to(dev)
            loss = (bce(sw, b["y_sw"].to(dev).float()) * m).sum() / m.sum().clamp(min=1)
            dm = b["dur_mask"].to(dev)
            if dm.any():
                y_dur = b["y_dur"].to(dev).reshape(-1)
                if ordinal:                  # one binary loss per cut, class-weighted like CE
                    dl = bce(dur.reshape(-1, N_DUR - 1), ordinal_targets(y_dur)).sum(-1)
                    if dur_w is not None:
                        dl = dl * dur_w[y_dur]
                else:
                    dl = ce(dur.reshape(-1, N_DUR), y_dur)
                loss = loss + args.dur_weight * ((dl * dm.reshape(-1)).sum() / dm.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.detach().item())
            step += 1
            if step % 200 == 0:
                print(f"  step {step:>5}/{args.steps}  loss {np.mean(running[-200:]):.4f}"
                      f"  {step/(time.time()-t0):.2f} it/s", flush=True)
            if args.eval_every and step % args.eval_every == 0 and step < args.steps:
                pt = curve_point(model, curve_loader, dev, args.eval_every_batches, step)
                curve.append(dict(pt, train_loss=float(np.mean(running[-args.eval_every:]))))
                print(f"  [val @ {step}] switch loss {pt['val_switch_loss']:.4f}  F1 {pt['val_f1']:.4f}"
                      f"  duration loss {pt['val_dur_loss'] if pt['val_dur_loss'] is None else round(pt['val_dur_loss'], 4)}"
                      f"  macro-F1 {pt['val_dur_macro_f1']:.4f}  ({pt['n']:,} positions)",
                      flush=True)
    print(f"trained {step} steps in {(time.time()-t0)/60:.1f} min")

    val = evaluate(model, val_loader, dev, args.eval_batches)
    tau, vf1 = tune_threshold(1 / (1 + np.exp(-val["logit"])), val["y"])
    print(f"threshold tuned on val: tau={tau:.3f} (val F1 {vf1:.3f})")

    test = evaluate(model, test_loader, dev, args.eval_batches)
    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        save_predictions(args.save_preds, val=val, test=test)
        print(f"saved predictions -> {args.save_preds}")
    report = score(test, tau)
    print("\n" + format_report(report, f"TEST split - {args.model} ({args.unit})"))
    if "shared" in report:
        print("\n" + format_report(report["shared"], "SHARED positions (last piece of each "
                                    "non-CJK word; comparable across tokenizers)"))
    payload = dict(model=args.model, unit=args.unit, **report, tau=tau, val_f1=vf1,
                   params=n_param, causality=chk, steps=step, args=vars(args))
    if curve:                             # --eval-every; the last point is the final model
        last = curve_point(model, curve_loader, dev, args.eval_every_batches, step)
        curve.append(dict(last, train_loss=float(np.mean(running[-args.eval_every:]))))
        payload["curve"] = curve
    if len(val["dtrue"]):                 # the duration decision, re-balanced on val
        dur_off, dvf = tune_class_offsets(val["dprob"], val["dtrue"])
        payload["duration_tuned"] = tuned_duration(test, tau, dur_off, dvf)
        print("\n" + format_duration(payload["duration_tuned"]["report"],
                                      f"TEST split, duration offsets tuned on val "
                                      f"(small, medium, large) {np.round(dur_off, 2).tolist()}:"))
    if holdout:
        zt = evaluate(model, loader("test", holdout), dev, args.eval_batches)
        zs = score(zt, tau)
        print("\n" + format_report(zs, "ZERO-SHOT - language pairs never seen in training"))
        payload["zero_shot"] = dict(**zs, holdout=holdout)
        if "duration_tuned" in payload and len(zt["dtrue"]):
            payload["zero_shot"]["duration_tuned"] = tuned_duration(zt, tau, dur_off, dvf)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save(dict(state=model.state_dict(), args=vars(args), tau=tau,
                    vocab_size=vocab_size, use_lang=use_lang), args.ckpt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out} and {args.ckpt}")


if __name__ == "__main__":
    main()
