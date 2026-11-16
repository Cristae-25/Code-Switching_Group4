"""Context-aware token classification via LinCE code-switch BERT models.

fastText scores each token in isolation, which is why it cannot resolve tokens
that are only disambiguated by their neighbours ("mas", "bila", "real"). These
are ``BertForTokenClassification`` models trained on the LinCE code-switch
benchmark, so they see the whole utterance.

Two models, routed by measured accuracy rather than by language family (see
``routing.MODEL_FOR``):

* ``hineng`` - the only thing that handles romanized Hinglish at all.
* ``spaeng`` - transfers to Italian/Malay/Tagalog as a generic
  Latin-L1-vs-English discriminator, but *regresses* German and French, which
  stay on fastText.

The ``ne`` (named-entity) label is mapped to ``other`` so proper nouns get
absorbed into the surrounding run instead of being forced into a language -
"Carney" and "Trump" are neither Hindi nor English.
"""
from __future__ import annotations

import sys
import time

import torch

from .routing import MODEL_FOR, MODEL_L1_LABEL, MODEL_REPO

_MAX_LEN = 512

# The registry lives on the `sys` module rather than here, deliberately.
# A module-level dict is wiped whenever IPython's %autoreload re-executes this
# file - which happens on *every* call in a notebook with autoreload on - so the
# 679 MB model would be reloaded from disk per row. `sys` is never reloaded, so
# hanging the cache there makes model loading genuinely once-per-session.
_REGISTRY = "_codeswitch_model_registry"


def _cache_dict() -> dict:
    reg = getattr(sys, _REGISTRY, None)
    if reg is None:
        reg = {}
        setattr(sys, _REGISTRY, reg)
    return reg


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_key: str, verbose: bool = True):
    """Load (once per session) a LinCE tokenizer/model pair onto the best device."""
    cache = _cache_dict()
    if model_key not in cache:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        repo = MODEL_REPO[model_key]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForTokenClassification.from_pretrained(repo).eval().to(_device())
        cache[model_key] = (tok, mdl)
        if verbose:
            print(f"[bert_tagger] loaded {model_key} on {_device()} "
                  f"in {time.time() - t0:.1f}s (once per session)", flush=True)
    return cache[model_key]


def diagnose() -> None:
    """Print why tagging might be slow. Run this if a cell takes minutes."""
    import transformers
    print(f"python      : {sys.executable}")
    print(f"transformers: {transformers.__version__}  (need >=5 with torch>=2.11)")
    print(f"torch       : {torch.__version__}")
    print(f"device      : {_device()}")
    cached = sorted(_cache_dict())
    print(f"models held : {cached or 'NONE - will load on first use'}")
    ip = sys.modules.get("IPython")
    if ip is not None and ip.get_ipython() is not None:
        exts = ip.get_ipython().extension_manager.loaded
        if "autoreload" in exts:
            print("autoreload  : ON - safe now (registry lives on sys), but it "
                  "wipes module globals; restart the kernel if you saw slowness")
        else:
            print("autoreload  : off")


def _label_words(model_key: str, words: list[str]) -> list[str]:
    """Raw LinCE label per word; first sub-token of each word decides."""
    tok, mdl = load(model_key)
    enc = tok(words, is_split_into_words=True, return_tensors="pt",
              truncation=True, max_length=_MAX_LEN)
    word_ids = enc.word_ids(0)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        pred = mdl(**enc).logits.argmax(-1)[0].tolist()
    out, seen = [], set()
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            seen.add(wid)
            out.append(mdl.config.id2label[pred[pos]])
    # Words past the 512-token limit are unlabelled; mark them unknown.
    return out + ["unk"] * (len(words) - len(out))


def tag_tokens_bert(tokens: list[str], first_language: str,
                    second_language: str) -> list[tuple[str, str]]:
    """Tag tokens with the LinCE model chosen for ``first_language``.

    Long token streams are processed in ``_MAX_LEN``-sized chunks so context is
    preserved within a chunk rather than lost per sentence.
    """
    key = MODEL_FOR[first_language]
    l1_label = MODEL_L1_LABEL[key]

    labels: list[str] = []
    # Leave headroom for [CLS]/[SEP] and sub-token expansion.
    step = _MAX_LEN // 3
    for start in range(0, len(tokens), step):
        labels.extend(_label_words(key, tokens[start:start + step]))

    out = []
    for tok, lab in zip(tokens, labels):
        if not any(ch.isalpha() for ch in tok):
            out.append((tok, "other"))
        elif lab == "en":
            out.append((tok, second_language))
        elif lab == l1_label:
            out.append((tok, first_language))
        else:
            # ne / ambiguous / mixed / fw / other / unk -> absorbed downstream.
            out.append((tok, "other"))
    return out
