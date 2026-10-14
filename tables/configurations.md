| Experiment | Data | GMR | Masking | Seeds | Epochs | Shown in |
|---|---|---|---|---|---|---|
| Main | 1°, 6 h | off, on | none, uniform 0–20 | 1 | 150 SS + 20 MS | `sec:results`, `appendix:additional_1deg_results` |
| Main | 4°, 1 day | off, on | none, uniform 0–20 | 5 | 150 SS | `appendix:additional_4deg_results` |
| Mask distribution | 4°, 1 day | off, on | Bernoulli p = 0.23 | 5 | 150 SS | `appendix:masking_ablations` |
| Mask sharing | 4°, 1 day | on | uniform 0–10 / 0–20, per batch / sample | 1 | 150 SS | `appendix:masking_ablations` |
| Mask indicator | 4°, 1 day | on | uniform 0–20, channels / FiLM / both | 1 | 150 SS | `appendix:masking_ablations` |
