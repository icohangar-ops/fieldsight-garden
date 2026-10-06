## Full test set (300 images)
| model | n | accuracy [95% CI] | macro-F1 | invalid | p50 latency | p95 latency | $ / 1k photos | cost vs FT |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B + LoRA (ours) | 300 | **79.7%** [75.0, 84.0] | 0.785 | 0.0% | 2.42 s | 3.38 s | $0.098 | 1.0× |
| Qwen3.6-35B-A3B base (zero-shot) | 300 | **67.7%** [62.3, 72.7] | 0.653 | 0.0% | 2.37 s | 3.53 s | $0.121 | 1.2× |
| Inkling (zero-shot, effort 0) | 300 | **49.7%** [44.0, 55.3] | 0.460 | 0.7% | 2.37 s | 3.55 s | $0.329 | 3.4× |

## Fixed 100-image subset (identical images for every model)
| model | n | accuracy [95% CI] | macro-F1 | invalid | p50 latency | p95 latency | $ / 1k photos | cost vs FT |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B + LoRA (ours) | 100 | **81.0%** [73.0, 88.0] | 0.786 | 0.0% | 2.42 s | 3.45 s | $0.099 | 1.0× |
| Qwen3.6-35B-A3B base (zero-shot) | 100 | **74.0%** [65.0, 82.0] | 0.713 | 0.0% | 2.36 s | 3.53 s | $0.122 | 1.2× |
| Inkling (zero-shot, effort 0) | 100 | **56.0%** [46.0, 65.0] | 0.503 | 0.0% | 2.39 s | 3.52 s | $0.333 | 3.4× |
| Inkling (zero-shot, effort 0.5) | 100 | **58.0%** [48.0, 67.0] | 0.516 | 0.0% | 3.35 s | 5.56 s | $0.755 | 7.6× |

## Paired accuracy difference vs fine-tuned (bootstrap 95% CI)
- full: Qwen3.6-35B-A3B base (zero-shot): FT minus model = +12.0 pts [+7.0, +17.0] (n=300)
- full: Inkling (zero-shot, effort 0): FT minus model = +30.0 pts [+23.7, +36.3] (n=300)
- subset100: Qwen3.6-35B-A3B base (zero-shot): FT minus model = +7.0 pts [+0.0, +14.0] (n=100)
- subset100: Inkling (zero-shot, effort 0): FT minus model = +25.0 pts [+15.0, +35.0] (n=100)
- subset100: Inkling (zero-shot, effort 0.5): FT minus model = +23.0 pts [+12.0, +34.0] (n=100)
