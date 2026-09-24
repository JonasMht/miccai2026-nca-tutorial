#!/usr/bin/env python3
"""Train a checkpoint and score it on the test split.

    python scripts/train_model.py --config full --epochs 240 --name liver

`--config fast` trains the notebook's session recipe instead. Writes
`checkpoints/ablation_cnca_<name>.pt` with the rollout length K and the vessel
threshold chosen on validation, plus `results.json` and `history.json` in
`checkpoints/ablation_cnca_<name>/`. Re-score any checkpoint later with
`scripts/score_checkpoint.py`.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ablation2d.data import load_splits  # noqa: E402
from ablation2d.evaluate import (fit_sphere_baseline, predict_all, report,  # noqa: E402
                                 sphere_baseline)
from ablation2d.model import best_vessel_threshold  # noqa: E402
from ablation2d.train import FAST, FULL, choose_inference_steps, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=("fast", "full"), default="full")
    ap.add_argument("--name", default=None, help="checkpoint name (default: the config)")
    ap.add_argument("--data", type=Path, default=ROOT / "data_liver")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--vessel-weight", type=float, default=None)
    ap.add_argument("--augment", action="store_true", help="D4 augmentation (off by default)")
    ap.add_argument("--free-conditioning", action="store_true",
                    help="do not rewrite the environment channels every step")
    ap.add_argument("--select", choices=("dice", "vessel_f1", "both"), default="dice",
                    help="validation score that picks the kept epoch")
    ap.add_argument("--no-cudnn", action="store_true",
                    help="for hosts whose cuDNN install fails on conv")
    ap.add_argument("--dry-run", action="store_true", help="print the config and exit")
    a = ap.parse_args()

    if a.no_cudnn:
        torch.backends.cudnn.enabled = False

    over = {"device": a.device, "augment_d4": a.augment,
            "restore_conditioning": not a.free_conditioning}
    for key, val in (("epochs", a.epochs), ("batch_size", a.batch_size), ("seed", a.seed),
                     ("channels", a.channels), ("vessel_weight", a.vessel_weight)):
        if val is not None:
            over[key] = val
    cfg = dataclasses.replace(FAST if a.config == "fast" else FULL, **over)
    print(f"[cfg ] {cfg}")
    if a.dry_run:
        return 0

    splits = load_splits(a.data, device=a.device)
    print("[data] " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))

    name = f"ablation_cnca_{a.name or a.config}"
    out = a.out / name
    model, history = train(cfg, splits["train"], splits["val"], out_dir=out, select=a.select)

    K, scores = choose_inference_steps(model, splits["val"])
    cfg = dataclasses.replace(cfg, infer_steps=K)
    print(f"[K   ] {K}, the cheapest within 1 % of the best: "
          f"{({k: round(v, 4) for k, v in scores.items()})}")

    sp = fit_sphere_baseline(splits["train"])
    print(f"[base] sphere r = {sp['C']:.4f} * P^{sp['ALPHA']:.4f} * t^{sp['BETA']:.4f}  ({sp['_fit']})")
    vthr, _ = best_vessel_threshold(predict_all(model, splits["val"], K), splits["val"].targets)
    print(f"[thr ] vessel threshold {vthr:.2f} (chosen on val)")

    one = report(model, splits["test"], steps=K, vessel_threshold=vthr,
                 baselines={"device-chart sphere": lambda x: sphere_baseline(x, sp)})
    avg = report(model, splits["test"], steps=K, vessel_threshold=vthr, average="repeats")
    res = {"C-NCA": one["C-NCA"], "C-NCA + 8 repeats": avg["C-NCA"],
           **{k: v for k, v in one.items() if k != "C-NCA"}}
    for n, r in res.items():
        if "dice" in r:
            line = (f"  {n:22s} DSC {r['dice']:.4f} (p10 {r['dice_p10']:.4f})  "
                    f"area err {r['area_err_cm2']:+.2f} cm2  "
                    f"recall {r['recall']:.3f}  precision {r['precision']:.3f}")
            if "vessel_f1" in r:
                line += f"  |  vessel F1 {r['vessel_f1']:.4f}"
            print(line)
        else:
            print(f"  {n:22s} vessel F1 {r['vessel_f1']:.4f}")

    (out / "results.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "infer_steps": K, "vessel_threshold": vthr,
         "step_sweep": {int(k): v for k, v in scores.items()},
         "sphere_baseline": sp, "test": res}, indent=1))
    # the checkpoint was saved before K and the threshold were known: store them
    ck = torch.load(out / "model.pt", weights_only=False)
    ck["config"]["infer_steps"] = K
    ck["vessel_threshold"] = vthr
    torch.save(ck, out / "model.pt")
    torch.save(ck, a.out / f"{name}.pt")
    print(f"[out ] {a.out / f'{name}.pt'}")


if __name__ == "__main__":
    main()
