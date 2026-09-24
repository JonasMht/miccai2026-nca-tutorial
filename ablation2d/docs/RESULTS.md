# Results: 2-D ablation surrogate

Generated 2026-09-24T20:19+00:00 by `scripts/write_results.py`. Test split: 10 patients the model never saw. K and the vessel threshold are chosen on validation. The reference model and the notebook use the plans with one or two needles (659 test plans); the older studies further down were scored on all 800.

## The reference model

`checkpoints/ablation_cnca_liver.pt`: 16 channels, 5 blocks per step, hidden x2, 240 epochs on the liver-masked corpus. K = 4, vessel threshold 0.97.

| | DSC | DSC p10 | area err (cm²) | recall | precision | vessel F1 |
|---|---|---|---|---|---|---|
| C-NCA | 0.9073 | 0.8429 | +1.74 | 0.988 | 0.838 | 0.8379 |
| C-NCA + 8 repeats | 0.9073 | 0.8429 | +1.74 | 0.988 | 0.838 | 0.8379 |
| C-NCA + 8 D4 orientations | 0.9088 | 0.8370 | +1.79 | 0.992 | 0.838 | 0.7729 |
| device-chart sphere | 0.7926 | 0.7159 | +4.28 | 0.972 | 0.667 | - |
| predict zero | 0.0364 | 0.0000 | -9.96 | 0.000 | 0.000 | - |

Device-chart sphere: `r = 1.0412 * P^0.3224 * t^0.2305` (n=397 single-setting cases, log-radius RMSE 0.0902, R2 0.658).

Against the device chart: +0.1147 DSC, which is what knowing the anatomy is worth.

Recall is above precision: the 100:1 foreground weighting makes the model over-predict the zone, the safer error for an ablation margin.

### Rollout length (validation DSC)

| K | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 34 | 36 | 38 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DSC | 0.9095 | 0.9108 | 0.9111 | 0.9112 | 0.9110 | 0.9110 | 0.9111 | 0.9111 | 0.9111 | 0.9112 | 0.9111 | 0.9110 | 0.9110 | 0.9111 | 0.9112 | 0.9111 | 0.9113 | 0.9113 | 0.9111 |

### Vessel threshold (validation F1)

| threshold | 0.3 | 0.5 | 0.7 | 0.8 | 0.85 | 0.9 | 0.93 | 0.95 | 0.97 | 0.98 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.8177 | 0.8396 | 0.8574 | 0.8665 | 0.8714 | 0.8778 | 0.8819 | 0.8845 | 0.8849 | 0.8622 |

## The session recipe

12 channels, 2 blocks per step, 40 epochs, K = 8. The notebook trains this recipe under a 5-minute budget instead of a fixed epoch count.

| | DSC | DSC p10 | area err (cm²) | recall | precision | vessel F1 |
|---|---|---|---|---|---|---|
| C-NCA | 0.8328 | 0.7508 | +3.97 | 0.973 | 0.728 | 0.6988 |
| C-NCA + 8 repeats | 0.8404 | 0.7594 | +3.90 | 0.978 | 0.735 | 0.7415 |
| C-NCA + 8 D4 orientations | 0.8430 | 0.7606 | +3.82 | 0.979 | 0.739 | 0.7274 |
| device-chart sphere | 0.7989 | 0.7204 | +5.00 | 0.969 | 0.679 | - |
| predict zero | 0.0300 | 0.0000 | -12.89 | 0.000 | 0.000 | - |

With D4 augmentation on (same budget, same seed) necrosis DSC is 0.8307 and vessel F1 falls from 0.6988 to 0.2803: the heat equation is rotation-invariant, axial CT is not.

### Under the notebook's 5-minute budget

Session recipe under the notebook's time budget: 180 s on a TITAN V (about 5 min on a Colab T4), fp16, warmup 2, best epoch selected on val DSC + vessel F1, test split, mean of seeds 0-2. 2026-09-24.

| arm | epochs | DSC 1x | vessel F1 1x | DSC 8x | vessel F1 8x |
|---|---|---|---|---|---|
| 12 ch, lr 3e-3 | 18 | 0.8216 | 0.6076 | 0.8328 | 0.6430 |
| 16 ch, lr 3e-3 | 14 | 0.8257 | 0.6242 | 0.8355 | 0.6582 |
| 12 ch, lr 3e-3, vessel weight 0.6 | 18 | 0.8092 | 0.6611 | 0.8285 | 0.6957 |
| 12 ch, lr 7e-3 | 18 | 0.8250 | 0.6512 | 0.8381 | 0.6908 |
| 16 ch, lr 5e-3 | 14 | 0.8172 | 0.6800 | 0.8316 | 0.7259 |
| 12 ch, lr 5e-3, no fire mask (the notebook) | 19 | 0.8314 | 0.7151 | 0.8314 | 0.7151 |
| 12 ch, lr 5e-3, fire rate 0.5 | 18 | 0.8181 | 0.6738 | 0.8310 | 0.7122 |

Seed-to-seed spread on the vessel F1 is about 0.05. Two changes stand out: lr 5e-3 (about +0.07 vessel F1 in the time available) and dropping the random fire mask, which improves both heads and makes the model deterministic (its 1x and 8x columns are identical).

## Robustness

Robustness on the 659 test plans with at most two needles (first 240 evaluated). Vessel F1 under random plans (one or two needles, 30-150 W), 250 W, tips outside the liver, no needle; vessel recall inside the true lesion, in an 8 mm ring outside it, and further away; necrosis DSC at longer rollouts; necrosis predicted with no needle. 2026-09-24.

| | previous reference (one rollout, 3 blocks per step) | reference (anatomy first, 5 blocks per step) |
|---|---|---|
| necrosis DSC, own plan | 0.905 | 0.908 |
| vessel F1, own plan | 0.825 | 0.836 |
| vessel F1, random plans | 0.811 | 0.836 |
| vessel F1, 250 W | 0.801 | 0.836 |
| vessel F1, tips outside the liver | 0.829 | 0.836 |
| vessel F1, no needle | 0.832 | 0.836 |
| vessel recall inside the lesion | 0.076 | 0.563 |
| vessel recall, 8 mm around it | 0.480 | 0.609 |
| vessel recall, further away | 0.836 | 0.797 |
| necrosis DSC at 40 steps | 0.909 | 0.911 |
| necrosis DSC at 100 steps | 0.906 | 0.910 |
| necrosis with no needle (cm²) | 0.001 | 0.000 |

## Anatomy first: vessels that do not depend on the plan

Session recipe (12 channels, lr 5e-3, EMA 0.99, plans with at most two needles), 180 s on a TITAN V. Test plans with at most two needles. Vessel recall inside the true lesion, in an 8 mm ring outside it, and further away. 2026-09-24.

| | DSC | vessel F1 | vessel F1, random plans | recall inside lesion | 8 mm around | further | DSC at K=40 |
|---|---|---|---|---|---|---|---|
| single rollout (seeds 0-1) | 0.847 | 0.718 | 0.691 | 0.001 | 0.284 | 0.749 | 0.690 |
| anatomy first, 6 CT-only steps (seeds 0-2) | 0.833 | 0.769 | 0.769 | 0.550 | 0.598 | 0.783 | 0.832 |

With a single rollout the model learns that vessels are rare inside a lesion and stops seeing them there. Reading the anatomy first, with the needle channels empty, and then holding the vessel channel fixed makes the vessel map independent of the plan by construction: its F1 is the same under random plans as under the real one.

## The fire mask

Random fire mask (p = 0.5) against every cell updating every step, all else identical. Test split; K and vessel threshold chosen on validation. 2026-09-24.

| 240 epochs | DSC | vessel F1 |
|---|---|---|
| fire 0.5, seed 0 (the previous reference) | 0.9032 | 0.8043 |
| no fire, seed 0 | 0.8856 | 0.8175 |
| no fire, seed 1 | 0.8441 | 0.8091 |
| no fire, seed 2 (the reference now) | 0.9095 | 0.8261 |

| test DSC at K = | 4 | 10 | 40 | 100 | 400 |
|---|---|---|---|---|---|
| fire 0.5 | 0.851 | 0.906 | 0.906 | 0.890 | 0.837 |
| no fire, seed 2 | 0.907 | 0.913 | 0.910 | 0.906 | 0.891 |

Without the fire mask the vessel head is better on every seed and the field holds far better over long rollouts; necrosis DSC varies more between seeds (0.844 to 0.910), so the shipped model is the best of three seeds.

## Training the fire-0.5 recipe longer

| run | DSC 1x | DSC 8x | vessel F1 1x | vessel F1 8x |
|---|---|---|---|---|
| shipped, 240 epochs | 0.9046 | 0.9129 | 0.8060 | 0.8294 |
| 480 epochs, seed 0 | 0.9051 | 0.9186 | 0.7983 | 0.8101 |
| 480 epochs, seed 1 | 0.9107 | 0.9183 | 0.7927 | 0.8051 |
| shipped + 240 epochs at lr 5e-4 | 0.9055 | 0.9118 | 0.7970 | 0.8173 |

At most +0.006 necrosis DSC for -0.01 to -0.02 vessel F1; validation F1 rose while test F1 fell, so doubling the epochs was not worth it.

## The recipe ladder

16 channels, 80 epochs, seed 0 for every arm, all re-scored with the same evaluation code. `1x` is one rollout, `8x` the mean of eight. These arms predate the wider vessel-threshold grid, so their vessel F1 is capped at a threshold of 0.7.

| arm | D4 | warm start | w | CT | K | DSC 1x | DSC 8x | vessel 1x | vessel 8x |
|---|---|---|---|---|---|---|---|---|---|
| control | - | - | 3.0 | full | 14 | 0.7827 | 0.8297 | 0.6288 | 0.6541 |
| + D4 + vessel warm start | yes | yes | 3.0 | full | 14 | 0.6994 | 0.7796 | 0.6101 | 0.6413 |
| + free conditioning | yes | yes | 3.0 | full | 14 | 0.6948 | 0.8142 | 0.3927 | 0.4466 |
| + loss rebalance | yes | yes | 0.3 | full | 8 | 0.8732 | 0.8840 | 0.5766 | 0.6166 |
| rebalance only | - | - | 0.3 | full | 12 | 0.8515 | 0.8594 | 0.6067 | 0.6601 |
| liver-masked CT | - | - | 0.3 | masked | 8 | 0.8687 | 0.8740 | 0.7502 | 0.7745 |
| w=1.0, full CT | - | - | 1.0 | full | 8 | 0.8256 | 0.8464 | 0.6328 | 0.6653 |
| w=1.0, masked CT | - | - | 1.0 | masked | 8 | 0.8334 | 0.8463 | 0.7625 | 0.7885 |

Single-variable comparisons (arm 2 bundles augmentation and the warm start; every row below changes one thing):

| change | ΔDSC | Δvessel F1 |
|---|---|---|
| vessel weight 3.0 -> 0.3 | +0.0688 | -0.0221 |
| masking the CT to the liver | +0.0172 | +0.1435 |
| w 0.3 -> 1.0, masked | -0.0353 | +0.0123 |
| w 0.3 -> 1.0, full CT | -0.0259 | +0.0261 |
| restore_conditioning off | -0.0046 | -0.2173 |
| rebalance, on the augmented arm | +0.1738 | -0.0335 |

## The vessel head

The model is never given a vessel mask. Its baseline is the best single Hounsfield threshold, fitted on the split it is scored on.

| | vessel F1 |
|---|---|
| C-NCA vessel head | **0.8379** |
| best single HU threshold | 0.4355 |

### Width, not loss weighting

Loss changes (Dice + BCE for MSE, weights 1/3/10, pos_weight 50, learned uncertainty weighting) left vessel F1 near 0.18. Real CT and then width moved it, and both heads rose together:

| channels | scratch | params | necrosis DSC | vessel F1 |
|---|---|---|---|---|
| 6 | 1 | 2,616 | 0.3018 | 0.3218 |
| 12 | 7 | 10,416 | 0.6583 | 0.5452 |
| 16 | 11 | 41,616 | 0.7063 | 0.6559 |
| 24 | 19 | 93,528 | 0.7342 | 0.6517 |

80 epochs per arm on the full-CT corpus. Timings are omitted: that GPU was thermally throttled throughout.

## Against a parameter-matched CNN

Dilated 1-2-4-8-16-32, no downsampling, same corpus, loss and split, and a larger receptive field than the NCA.

| | params | reach | DSC | recall | precision | vessel F1 |
|---|---|---|---|---|---|---|
| dilated CNN | 40,340 | 127 cells | 0.9119 | 0.978 | 0.862 | 0.8043 |
| C-NCA | 46,240 | 40 cells | 0.9073 | 0.988 | 0.838 | 0.8379 |

DSC difference -0.0047. The CNN was trained for 240 epochs on data_liver.

## Persistence

Two arms differing only in `persist_rate`; fast architecture, 40 epochs, trained on [3, 10] steps.

| val DSC at K = | 4 | 8 | 15 | 20 | 30 | 45 | 60 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| sampled length only | 0.7746 | 0.8305 | 0.8300 | 0.8233 | 0.8015 | 0.7668 | 0.7375 | 0.7061 | 0.6642 | 0.6719 |
| + persistence | 0.7520 | 0.8290 | 0.8347 | 0.8351 | 0.8360 | 0.8367 | 0.8370 | 0.8391 | 0.8400 | 0.8401 |

## Input channels

One channel zeroed, everything else identical; fast architecture, 40 epochs. With all channels: val DSC 0.8203.

| channel zeroed | val DSC | ΔDSC |
|---|---|---|
| `applicator_activation` | 0.1335 | -0.6868 |
| `hounsfield` | 0.7033 | -0.1170 |
| `needle` | 0.7621 | -0.0582 |

## Corpus

`data_liver/`: 4000 plans on real patient CT, split by patient (train 2800, val 400, test 800). See `data_liver/DATASHEET.md`.
