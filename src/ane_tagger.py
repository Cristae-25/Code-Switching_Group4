"""AnE-LID: one word-level tagger for every "<L1> vs English" pair.

``igorsterner/AnE-LID`` (Sterner, VarDial 2024) is XLM-R-large fine-tuned on
four English code-switch pairs at once - LinCE hineng/spaeng/nepeng plus
Denglisch - with the labels ``English`` / ``notEnglish`` / ``Mixed`` /
``Other``. Training on several pairs is what lets it transfer to pairs it never
saw, where the single-pair LinCE models (and our XLM-R hineng fine-tune) do
not. Licence: academic use only.

It only answers "English or not", so it cannot tag the German<->French rows.

The companion ``igorsterner/AnE-NER`` (same base, labels ``I`` / ``O``) marks
named-entity words; ``entity_mask_batch`` exposes it for the NE pass.

``Mixed`` (an English stem inside L1 morphology, "rewatchen") maps to the first
language - the word's grammatical host. ``Other`` maps to ``other``.

Word probabilities are the mean of the word's sub-token softmaxes, which is the
model card's majority vote made continuous, so a confidence comes for free.
"""
from __future__ import annotations

import sys

import torch

from .routing import ane_to_language as to_language  # noqa: F401  (public name)

REPO = "igorsterner/AnE-LID"
NER_REPO = "igorsterner/AnE-NER"
_REGISTRY = "_codeswitch_ane_registry"   # on `sys` for the same reason as bert_tagger
_CHUNK_WORDS = 170                       # headroom under 512 sub-tokens


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(repo: str = REPO, device: str | None = None):
    """Load (once per session) ``repo`` onto ``device`` (default: best available).

    On CUDA the model runs in fp16, which roughly doubles throughput; on MPS it
    measured no faster, and on CPU it is unsupported, so both stay fp32. Masks
    agreed with fp32 on 99.99% of tokens.
    """
    reg = getattr(sys, _REGISTRY, None)
    if reg is None:
        reg = {}
        setattr(sys, _REGISTRY, reg)
    dev = device or _device()
    key = (repo, dev)                  # one copy per GPU, so passes can shard
    if key not in reg:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForTokenClassification.from_pretrained(repo).eval()
        if dev.startswith("cuda"):
            mdl = mdl.half()
        reg[key] = (tok, mdl.to(dev))
    return reg[key]


def entity_mask_batch(sentences: list[list[str]],
                      device: str | None = None) -> list[list[bool]]:
    """True where AnE-NER calls the word part of a named entity."""
    return [[lab == "I" for lab, _ in row]
            for row in word_probs_batch(sentences, repo=NER_REPO, device=device)]


def word_probs_batch(sentences: list[list[str]], repo: str = REPO,
                     device: str | None = None) -> list[list[tuple[str, float]]]:
    """``(label, confidence)`` per word, for a batch of pre-tokenised sentences.

    Sentences longer than ``_CHUNK_WORDS`` are split into chunks, so no word is
    ever truncated away; context is preserved within a chunk.
    """
    tok, mdl = load(repo, device)
    chunks, owner = [], []
    for si, words in enumerate(sentences):
        for start in range(0, max(len(words), 1), _CHUNK_WORDS):
            chunks.append(list(words[start:start + _CHUNK_WORDS]))
            owner.append(si)

    enc = tok(chunks, is_split_into_words=True, return_tensors="pt",
              padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        probs = mdl(**{k: v.to(mdl.device) for k, v in enc.items()}).logits.float().softmax(-1).cpu()

    out: list[list[tuple[str, float]]] = [[] for _ in sentences]
    id2label = mdl.config.id2label
    for ci, words in enumerate(chunks):
        acc = [[] for _ in words]
        for pos, wid in enumerate(enc.word_ids(ci)):
            if wid is not None:
                acc[wid].append(probs[ci, pos])
        for a in acc:
            if not a:
                out[owner[ci]].append(("Other", 0.0))
                continue
            p = torch.stack(a).mean(0)
            out[owner[ci]].append((id2label[int(p.argmax())], float(p.max())))
    return out




def labels_batch(sentences: list[list[str]], device: str | None = None) -> list[list[str]]:
    """Raw AnE-LID label per word - what ``persist.ane_labels`` caches."""
    return [[lab for lab, _ in row] for row in word_probs_batch(sentences, device=device)]


def tag_tokens_ane(tokens: list[str], first_language: str,
                   second_language: str = "English",
                   labels: list[str] | None = None) -> list[tuple[str, str]]:
    """Same contract as the other taggers: ``(token, language)`` pairs.

    Pass ``labels`` (from :func:`labels_batch`) to skip the model call.
    """
    if labels is None or len(labels) != len(tokens):
        labels = labels_batch([tokens])[0]
    return [(t, "other" if not any(c.isalpha() for c in t)
             else to_language(lab, first_language, second_language))
            for t, lab in zip(tokens, labels)]
