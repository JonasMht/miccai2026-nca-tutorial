#!/usr/bin/env python3
"""Score a checkpoint with the current evaluation code.

    python3 scripts/score_checkpoint.py checkpoints/ablation_cnca_liver.pt

A training run's own results.json was written by the code as it was when the
run started, so arms trained across an evaluation change are only comparable
after re-scoring. Everything is chosen on validation and reported on test: K
(cheapest within 1 % of the best), the vessel threshold, and the sphere
baseline (fitted on train). Writes `<checkpoint>.scored.json`.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ablation2d.data import load_splits  # noqa: E402
from ablation2d.evaluate import (fit_sphere_baseline, report,  # noqa: E402
                                 sphere_baseline)
from ablation2d.model import best_vessel_threshold  # noqa: E402
from ablation2d.train import (choose_inference_steps,  # noqa: E402
                              load_checkpoint, predict_d4_tta,
                              predict_repeats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--data", type=Path, default=ROOT / "data_liver")
    ap.add_argument("--max-needles", type=int, default=None,
                    help="score only plans with at most this many needles")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the JSON (default: next to the ckpt)")
    a = ap.parse_args()

    model, cfg, meta = load_checkpoint(a.ckpt, device=a.device)
    splits = load_splits(a.data, device=a.device, max_needles=a.max_needles)
    print(f"[ckpt] {a.ckpt}  channels={cfg.channels} sub={cfg.n_sub_models} "
          f"epochs={cfg.epochs} augment_d4={cfg.augment_d4}")
    if meta.get("config_defaulted"):
        print(f"[warn] this checkpoint predates "
              f"{', '.join(meta['config_defaulted'])} - those values above are "
              f"today's defaults, not what the run used.")

    K, scores, vscores = choose_inference_steps(model, splits["val"],
                                                return_vessel=True,
                                                batch_size=16)
    if vscores:
        kv = min(k for k, v in vscores.items()
                 if v >= max(vscores.values()) * 0.99)
        print(f"[K   ] necrosis wants K={K}, the vessel head wants K={kv}"
              + ("" if kv == K else
                 f"; reporting at K={K}: vessel F1 {vscores[K]:.4f} there, "
                 f"{vscores[kv]:.4f} at K={kv}"))
    # batched: a whole split at once runs out of memory on a small GPU
    def _predict(ds, mode, bs=16):
        outs = []
        for x, _ in ds.batches(bs, shuffle=False):
            if mode == "d4":
                outs.append(predict_d4_tta(model, x, K))
            elif mode == "repeats":
                outs.append(predict_repeats(model, x, K))
            else:
                with torch.no_grad():
                    outs.append(model(x, steps=K))
        return torch.cat(outs)

    pv_val = _predict(splits["val"], "none")
    vthr, vf1_val = best_vessel_threshold(pv_val, splits["val"].targets)
    from ablation2d.model import vessel_f1
    thr_curve = {round(t, 2): round(float(vessel_f1(pv_val, splits["val"].targets, t).mean()), 4)
                 for t in (0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98)}
    print(f"[val ] K={K}   vessel threshold {vthr:.2f} (val F1 {vf1_val:.4f})")

    sp = fit_sphere_baseline(splits["train"])
    # Three test-time reductions: one rollout, the mean of 8, the mean over the
    # 8 orientations. Without "repeats", D4 would be credited with the variance
    # reduction that any averaging gives. The threshold is re-tuned per mode.
    def scored(mode):
        pv = _predict(splits["val"], mode)
        t, _ = best_vessel_threshold(pv, splits["val"].targets)
        r = report(model, splits["test"], steps=K, vessel_threshold=t,
                   average=mode,
                   baselines=({"device-chart sphere":
                               lambda x: sphere_baseline(x, sp)}
                              if mode == "none" else None))
        r["C-NCA"]["vessel_threshold"] = t
        return r

    base = scored("none")
    res = {"C-NCA": base["C-NCA"],
           "C-NCA + 8 repeats": scored("repeats")["C-NCA"],
           "C-NCA + 8 D4 orientations": scored("d4")["C-NCA"],
           **{k: v for k, v in base.items() if k != "C-NCA"}}

    for name, r in res.items():
        if "dice" in r:
            line = (f"  {name:22s} DSC {r['dice']:.4f} (p10 {r['dice_p10']:.4f})"
                    f"  area err {r['area_err_cm2']:+.2f} cm2  "
                    f"recall {r['recall']:.3f}  precision {r['precision']:.3f}")
            if "vessel_f1" in r:
                line += f"  |  vessel F1 {r['vessel_f1']:.4f}"
            print(line)
        else:
            print(f"  {name:22s} vessel F1 {r['vessel_f1']:.4f}")

    out = a.out or a.ckpt.with_suffix(".scored.json")
    out.write_text(json.dumps(
        {"checkpoint": str(a.ckpt), "infer_steps": K, "max_needles": a.max_needles,
         "vessel_threshold": vthr, "vessel_threshold_curve_val": thr_curve,
         "step_sweep": {int(k): v for k, v in scores.items()},
         "step_sweep_vessel": {int(k): v for k, v in vscores.items()},
         "sphere_baseline": sp, "test": res,
         "config": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(cfg).items()},
         "config_defaulted": meta.get("config_defaulted", [])}, indent=1))
    print(f"[out ] {out}")


if __name__ == "__main__":
    sys.exit(main())
