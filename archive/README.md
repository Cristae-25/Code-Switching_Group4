# archive

Retired experiments, kept so the negative results in `docs/labelling_pipeline.md` can be reproduced. None of them is part of the live pipeline. Run them from the repo root, e.g. `./.venv/bin/python archive/eval_dict.py`.

| File | What it produced |
|---|---|
| `data_pipeline_plan.md` | The original plan; superseded in several places |
| `eval_dict.py` | dict-naive / dict-freq / fastText / mBERT on the gold set (§3.6–3.7) |
| `train_xlmr_lid.py`, `eval_xlmr.py` | XLM-R hineng fine-tune and its failed transfer (§3.5) |
| `train_canine_lid.py`, `eval_canine.py` | CANINE character-level model, respelling stress test (§3.8) |

Their model weights were deleted to save disk space. Retrain XLM-R (about 110 min) and CANINE; re-download fastText's `models/lid.176.bin` (the URL is in `src/model_tagger.py`).
