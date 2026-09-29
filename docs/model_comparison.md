# Model comparison (test split)

Every run is scored the same way (`src/evaluation.py`): the switch threshold is tuned on val and applied unchanged to test; duration is scored at true switch points whose class is certain. **Words** and **pieces** are different units (a word is 1.3–1.7 pieces), so compare models within a unit.

## Overall

| Model | Switch F1 | Macro-F1 over pairs | F1 into L2 | F1 into L1 | Duration acc | Duration macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.295 | 0.259 | 0.000 | 0.382 | 0.585 | 0.428 |
| hazard, per pair (words) | 0.329 | 0.305 | 0.227 | 0.393 | 0.585 | 0.428 |
| hazard, pooled (pieces) | 0.235 | 0.214 | 0.000 | 0.289 | 0.509 | 0.225 |
| hazard, per pair (pieces) | 0.266 | 0.259 | 0.217 | 0.299 | 0.509 | 0.225 |
| scratch (word) | 0.513 | 0.512 | 0.350 | 0.655 | 0.712 | 0.571 |
| scratch (subword) | 0.453 | 0.453 | 0.356 | 0.560 | 0.636 | 0.482 |
| xlmr (subword) | 0.646 | 0.676 | 0.536 | 0.759 | 0.678 | 0.575 |

## Switch F1 by language pair

Pairs are sorted from easiest to hardest (mean F1 across the models shown). **Bold** marks the best model for each pair.

| Pair | hazard, pooled (words) | hazard, per pair (words) | hazard, pooled (pieces) | hazard, per pair (pieces) | scratch (word) | scratch (subword) | xlmr (subword) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hindi->English | 0.381 | 0.442 | 0.329 | 0.362 | 0.626 | 0.541 | **0.715** |
| Korean->English | 0.247 | 0.252 | 0.207 | 0.265 | 0.639 | 0.611 | **0.808** |
| Philippines->English | 0.313 | 0.394 | 0.264 | 0.323 | 0.557 | 0.492 | **0.638** |
| German->English | 0.248 | 0.318 | 0.210 | 0.268 | 0.528 | 0.469 | **0.733** |
| Japanese->English | 0.232 | 0.244 | 0.190 | 0.243 | 0.552 | 0.527 | **0.736** |
| Cantonese->English | 0.327 | 0.328 | 0.249 | 0.252 | 0.500 | 0.442 | **0.619** |
| French->English | 0.233 | 0.302 | 0.195 | 0.208 | 0.509 | 0.453 | **0.695** |
| Arabic->English | 0.239 | 0.322 | 0.197 | 0.241 | 0.486 | 0.412 | **0.664** |
| Malay->English | 0.277 | 0.328 | 0.241 | 0.281 | 0.470 | 0.397 | **0.548** |
| Spanish->English | 0.232 | 0.278 | 0.191 | 0.239 | 0.498 | 0.425 | **0.673** |
| Russian->English | 0.236 | 0.311 | 0.199 | 0.243 | 0.477 | 0.387 | **0.659** |
| Vietnamese->English | 0.267 | 0.270 | 0.222 | 0.248 | 0.459 | 0.411 | **0.622** |
| Italian->English | 0.227 | 0.285 | 0.192 | 0.242 | 0.476 | 0.406 | **0.671** |
| Chinese->English | 0.221 | 0.225 | 0.179 | 0.242 | 0.465 | 0.429 | **0.699** |
| German->French | 0.198 | 0.280 | 0.150 | 0.223 | 0.435 | 0.387 | **0.658** |

Spread between the easiest and hardest pair: hazard, pooled (words) 0.183, hazard, per pair (words) 0.217, hazard, pooled (pieces) 0.179, hazard, per pair (pieces) 0.154, scratch (word) 0.204, scratch (subword) 0.224, xlmr (subword) 0.260.
