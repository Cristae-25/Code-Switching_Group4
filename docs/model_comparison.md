# Model comparison (test split)

Every run is scored the same way (`src/evaluation.py`): the switch threshold is tuned on val and applied unchanged to test; duration is scored at true switch points whose class is certain. **Words** and **pieces** are different units (a word is 1.3–1.7 pieces), so compare models within a unit.

## Overall

**Shared F1** is scored only at the last piece of each non-CJK word, positions that every tokenizer (and the word model) has in common. It is the fair comparison between models with different tokenizers. It is empty for runs made before it existed.

| Model | Switch F1 | Macro-F1 over pairs | Shared F1 (non-CJK words) | F1 into L2 | F1 into L1 | Duration acc | Duration macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.295 | 0.259 | 0.345 | 0.000 | 0.382 | 0.585 | 0.428 |
| hazard, per pair (words) | 0.329 | 0.305 | 0.373 | 0.227 | 0.393 | 0.585 | 0.428 |
| hazard, pooled (pieces) | 0.235 | 0.214 | 0.332 | 0.000 | 0.289 | 0.509 | 0.225 |
| hazard, per pair (pieces) | 0.266 | 0.259 | 0.360 | 0.217 | 0.299 | 0.509 | 0.225 |
| hazard, per pair (RoBERTa pieces) | 0.264 | 0.224 | 0.338 | 0.025 | 0.327 | 0.627 | 0.446 |
| hazard, per pair (mBERT pieces) | 0.254 | 0.219 | 0.357 | 0.159 | 0.307 | 0.552 | 0.402 |
| scratch (word) | 0.508 | 0.505 | 0.588 | 0.342 | 0.651 | 0.711 | 0.570 |
| scratch (subword) | 0.451 | 0.451 | 0.546 | 0.354 | 0.557 | 0.637 | 0.482 |
| xlmr (subword) | 0.646 | 0.676 |  | 0.536 | 0.759 | 0.678 | 0.575 |
| xlmr (subword, trained on 10 pairs) | 0.634 | 0.656 |  | 0.502 | 0.764 | 0.674 | 0.574 |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.625 | 0.627 |  | 0.529 | 0.730 | 0.660 | 0.562 |
| mbert (subword) | 0.634 | 0.662 | 0.719 | 0.512 | 0.759 | 0.713 | 0.576 |
| scratch (subword, trained on 10 pairs) | 0.447 | 0.435 |  | 0.331 | 0.565 | 0.632 | 0.481 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.330 | 0.322 |  | 0.205 | 0.500 | 0.568 | 0.398 |
| naive_l2 (6 script pairs, pieces) | 0.225 | 0.183 | 0.323 | 0.000 | 0.259 |  |  |
| piece lookup (6 script pairs) | 0.427 | 0.450 | 0.567 | 0.253 | 0.547 | 0.628 | 0.473 |

## Switch F1 by language pair

Pairs are sorted from easiest to hardest (mean F1 across the models shown). **Bold** marks the best model for each pair.

| Pair | hazard, pooled (words) | hazard, per pair (words) | hazard, pooled (pieces) | hazard, per pair (pieces) | hazard, per pair (RoBERTa pieces) | hazard, per pair (mBERT pieces) | scratch (word) | scratch (subword) | xlmr (subword) | xlmr (subword, trained on 10 pairs) | xlmr (subword, trained on 10 pairs, held-out pairs) | mbert (subword) | scratch (subword, trained on 10 pairs) | scratch (subword, trained on 10 pairs, held-out pairs) | naive_l2 (6 script pairs, pieces) | piece lookup (6 script pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hindi->English | 0.381 | 0.442 | 0.329 | 0.362 | 0.355 | 0.307 | 0.616 | 0.544 | **0.715** |  | 0.595 | 0.706 |  | 0.372 |  |  |
| Philippines->English | 0.313 | 0.394 | 0.264 | 0.323 | 0.290 | 0.287 | 0.548 | 0.494 | 0.638 | **0.639** |  | 0.627 | 0.498 |  |  |  |
| Korean->English | 0.247 | 0.252 | 0.207 | 0.265 | 0.232 | 0.208 | 0.636 | 0.626 | **0.808** |  | 0.775 | 0.793 |  | 0.318 | 0.177 | 0.610 |
| German->English | 0.248 | 0.318 | 0.210 | 0.268 | 0.236 | 0.248 | 0.516 | 0.464 | **0.733** |  | 0.653 | 0.718 |  | 0.359 |  |  |
| Japanese->English | 0.232 | 0.244 | 0.190 | 0.243 | 0.218 | 0.182 | 0.537 | 0.522 | 0.736 | **0.737** |  | 0.700 | 0.516 |  | 0.156 | 0.495 |
| French->English | 0.233 | 0.302 | 0.195 | 0.208 | 0.225 | 0.195 | 0.518 | 0.458 | **0.695** | 0.694 |  | 0.689 | 0.430 |  |  |  |
| Cantonese->English | 0.327 | 0.328 | 0.249 | 0.252 | 0.277 | 0.265 | 0.498 | 0.439 | 0.619 | **0.623** |  | 0.609 | 0.449 |  | 0.244 | 0.417 |
| Spanish->English | 0.232 | 0.278 | 0.191 | 0.239 | 0.212 | 0.223 | 0.489 | 0.425 | **0.673** | 0.669 |  | 0.660 | 0.422 |  |  |  |
| Italian->English | 0.227 | 0.285 | 0.192 | 0.242 | 0.214 | 0.193 | 0.467 | 0.414 | **0.671** |  | 0.661 | 0.658 |  | 0.366 |  |  |
| Vietnamese->English | 0.267 | 0.270 | 0.222 | 0.248 | 0.232 | 0.221 | 0.453 | 0.403 | 0.622 | **0.623** |  | 0.608 | 0.410 |  |  |  |
| Malay->English | 0.277 | 0.328 | 0.241 | 0.281 | 0.232 | 0.254 | 0.459 | 0.397 | **0.548** | 0.545 |  | 0.531 | 0.396 |  |  |  |
| Arabic->English | 0.239 | 0.322 | 0.197 | 0.241 | 0.224 | 0.176 | 0.482 | 0.400 | 0.664 | **0.666** |  | 0.649 | 0.406 |  | 0.175 | 0.390 |
| Russian->English | 0.236 | 0.311 | 0.199 | 0.243 | 0.220 | 0.188 | 0.470 | 0.372 | 0.659 | **0.667** |  | 0.645 | 0.395 |  | 0.183 | 0.391 |
| Chinese->English | 0.221 | 0.225 | 0.179 | 0.242 | 0.199 | 0.142 | 0.448 | 0.435 | **0.699** | 0.697 |  | 0.676 | 0.423 |  | 0.161 | 0.399 |
| German->French | 0.198 | 0.280 | 0.150 | 0.223 | 0.000 | 0.202 | 0.433 | 0.368 | 0.658 |  | 0.450 | **0.663** |  | 0.194 |  |  |

## Shared positions by language pair

Switch F1 on the last piece of each non-CJK word (the same words for every model).

| Pair | hazard, pooled (words) | hazard, per pair (words) | hazard, pooled (pieces) | hazard, per pair (pieces) | hazard, per pair (RoBERTa pieces) | hazard, per pair (mBERT pieces) | scratch (word) | scratch (subword) | mbert (subword) | naive_l2 (6 script pairs, pieces) | piece lookup (6 script pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cantonese->English | 0.434 | 0.427 | 0.414 | 0.400 | 0.384 | 0.406 | 0.675 | 0.622 | **0.791** | 0.364 | 0.611 |
| Hindi->English | 0.381 | 0.442 | 0.373 | 0.434 | 0.401 | 0.433 | 0.616 | 0.586 | **0.718** |  |  |
| Korean->English | 0.311 | 0.329 | 0.305 | 0.318 | 0.322 | 0.313 | 0.663 | 0.584 | **0.803** | 0.240 | 0.539 |
| Philippines->English | 0.313 | 0.394 | 0.310 | 0.391 | 0.342 | 0.393 | 0.548 | 0.528 | **0.637** |  |  |
| Japanese->English | 0.281 | 0.306 | 0.277 | 0.273 | 0.296 | 0.267 | 0.616 | 0.560 | **0.770** | 0.210 | 0.469 |
| German->English | 0.248 | 0.318 | 0.241 | 0.322 | 0.248 | 0.319 | 0.516 | 0.500 | **0.728** |  |  |
| Chinese->English | 0.269 | 0.284 | 0.260 | 0.189 | 0.270 | 0.235 | 0.608 | 0.577 | **0.773** | 0.218 | 0.470 |
| Malay->English | 0.277 | 0.328 | 0.275 | 0.321 | 0.265 | 0.314 | 0.459 | 0.419 | **0.537** |  |  |
| French->English | 0.233 | 0.302 | 0.223 | 0.238 | 0.235 | 0.224 | 0.518 | 0.494 | **0.696** |  |  |
| Spanish->English | 0.232 | 0.278 | 0.222 | 0.281 | 0.225 | 0.270 | 0.489 | 0.457 | **0.666** |  |  |
| Vietnamese->English | 0.267 | 0.271 | 0.258 | 0.270 | 0.254 | 0.257 | 0.453 | 0.413 | **0.613** |  |  |
| Arabic->English | 0.239 | 0.322 | 0.229 | 0.311 | 0.235 | 0.197 | 0.482 | 0.431 | **0.663** | 0.204 | 0.412 |
| Italian->English | 0.227 | 0.285 | 0.220 | 0.292 | 0.223 | 0.221 | 0.467 | 0.446 | **0.664** |  |  |
| Russian->English | 0.236 | 0.311 | 0.228 | 0.307 | 0.231 | 0.214 | 0.470 | 0.408 | **0.653** | 0.212 | 0.410 |
| German->French | 0.198 | 0.280 | 0.190 | 0.272 | 0.000 | 0.267 | 0.433 | 0.403 | **0.673** |  |  |

## Duration macro-F1 by language pair

At true switch points whose class is certain. CJK runs count characters (or merged pieces), other scripts count words, so compare within a script.

| Pair | piece lookup (6 script pairs) |
|---|---:|
| Cantonese->English | **0.479** |
| Japanese->English | **0.464** |
| Korean->English | **0.431** |
| Chinese->English | **0.405** |
| Arabic->English | **0.400** |
| Russian->English | **0.396** |

Spread between the easiest and hardest pair: hazard, pooled (words) 0.183, hazard, per pair (words) 0.217, hazard, pooled (pieces) 0.179, hazard, per pair (pieces) 0.154, hazard, per pair (RoBERTa pieces) 0.355, hazard, per pair (mBERT pieces) 0.166, scratch (word) 0.204, scratch (subword) 0.258, xlmr (subword) 0.260, xlmr (subword, trained on 10 pairs) 0.192, xlmr (subword, trained on 10 pairs, held-out pairs) 0.325, mbert (subword) 0.263, scratch (subword, trained on 10 pairs) 0.121, scratch (subword, trained on 10 pairs, held-out pairs) 0.178, naive_l2 (6 script pairs, pieces) 0.089, piece lookup (6 script pairs) 0.220.
