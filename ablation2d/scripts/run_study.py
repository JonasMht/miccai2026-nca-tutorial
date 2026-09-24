#!/usr/bin/env python3
"""The comparative measurements the talk quotes.

    python scripts/run_study.py --all --out results

* `--baseline`   a parameter-matched dilated CNN, same corpus, loss and split;
* `--stability`  how far past its training range the rollout holds, with and
                 without persistence training;
* `--ablation`   zero each input channel in turn and retrain.

Each study writes one JSON in results/, which the slide builder reads.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d.baseline_cnn import DilatedCNN, match_params  # noqa: E402
from ablation2d.channels import CHANNEL_NAMES  # noqa: E402
from ablation2d.data import load_splits  # noqa: E402
from ablation2d.evaluate import predict_all, score  # noqa: E402
from ablation2d.model import AblationCNCA, best_vessel_threshold, vessel_f1  # noqa: E402
from ablation2d.train import FAST, FULL, evaluate, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _cfg(device, arch="full", **over):
    """The study config. The comparative studies run the session ("fast")
    architecture; the CNN baseline always matches the reference model."""
    base = FAST if arch == "fast" else FULL
    return dataclasses.replace(base, device=device, **over)


# --------------------------------------------------------------------------- #
def study_baseline(splits, device, out: Path, epochs: int):
    """NCA against a parameter-matched, longer-reaching dilated CNN."""
    nca_ref = AblationCNCA()
    width = match_params(nca_ref.n_params)
    cnn = DilatedCNN(width, out_channels=2)
    print(f"[baseline] NCA {nca_ref.n_params:,} params (reach "
          f"{nca_ref.reach_cells * FULL.infer_steps} cells at K={FULL.infer_steps}) "
          f"vs CNN {cnn.n_params:,} params (reach {cnn.reach_cells} cells)")

    # no rollout, so the rollout tricks have nothing to act on
    cfg = _cfg(device, epochs=epochs, step_min=1, step_max=1, infer_steps=1,
               further_steps_rate=0.0, persist_rate=0.0, batch_size=16)
    t0 = time.time()
    cnn, _ = train(cfg, splits["train"], splits["val"], model=cnn,
                   log=lambda s: print("   " + s))
    with torch.no_grad():
        thr, _ = best_vessel_threshold(predict_all(cnn, splits["val"], 1), splits["val"].targets)
        pt = predict_all(cnn, splits["test"], 1)
    cnn_res = score(pt, splits["test"].cell_death)
    cnn_res["vessel_f1"] = float(vessel_f1(pt, splits["test"].targets, thr).mean())
    cnn_res["vessel_threshold"] = thr
    cnn_res["epochs"] = epochs
    cnn_res["params"] = cnn.n_params
    cnn_res["reach_cells"] = cnn.reach_cells
    cnn_res["train_min"] = round((time.time() - t0) / 60, 1)
    print(f"[baseline] CNN  DSC {cnn_res['dice']:.4f}  "
          f"recall {cnn_res['recall']:.3f}  precision {cnn_res['precision']:.3f}")
    (out / "baseline_cnn.json").write_text(json.dumps(cnn_res, indent=1))
    torch.save({"config": dataclasses.asdict(cfg), "width": width,
                "state_dict": cnn.state_dict()},
               ROOT / "checkpoints" / "baseline_cnn.pt")
    return cnn_res


# --------------------------------------------------------------------------- #
def study_stability(splits, device, out: Path, epochs: int, arch="fast",
                    ks=(4, 8, 15, 20, 30, 45, 60, 100, 200, 400)):
    """Does persistence training actually hold the field? Two models, one flag."""
    res = {"_arch": arch, "_epochs": epochs}
    for tag, persist in (("with_persistence", 0.5), ("without_persistence", 0.0)):
        cfg = _cfg(device, arch, epochs=epochs, persist_rate=persist, batch_size=16)
        m, _ = train(cfg, splits["train"], splits["val"], log=lambda s: None)
        row = {}
        for k in ks:
            _, d = evaluate(m, splits["val"], steps=k, batch_size=16)
            row[k] = round(d, 4)
        # the persistence range decides this experiment, so it goes in the file
        res[tag] = {"trained_steps": [cfg.step_min, cfg.step_max],
                    "persist_range": list(cfg.persist_range),
                    "persist_rate": cfg.persist_rate,
                    "persist_bptt": cfg.persist_bptt,
                    "dice_by_k": row}
        print(f"[stability] {tag:20s} " +
              "  ".join(f"K={k}:{v:.3f}" for k, v in row.items()))
        torch.save({"config": dataclasses.asdict(cfg), "state_dict": m.state_dict()},
                   ROOT / "checkpoints" / f"stability_{tag}.pt")
    (out / "stability.json").write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- #
ABLATION_CHANNELS = ("hounsfield", "needle", "applicator_activation")


def study_channel_ablation(splits, device, out: Path, epochs: int, arch="fast",
                           channels=ABLATION_CHANNELS, seeds: int = 1):
    """Zero one input channel and retrain. Does it earn its place?

    The channel is zeroed rather than removed, so the architecture is unchanged
    and only the information differs.
    """
    res = {"_arch": arch, "_epochs": epochs, "_seeds": seeds}
    raw = {}
    for drop in [None] + [CHANNEL_NAMES.index(c) for c in channels]:
        name = "all channels" if drop is None else f"drop {CHANNEL_NAMES[drop]}"
        scores = []
        for seed in range(seeds):
            cfg = _cfg(device, arch, epochs=epochs, batch_size=16, seed=seed)

            class Masked(AblationCNCA):
                def forward(self, inputs, *a, **kw):
                    if drop is not None:
                        inputs = inputs.clone()
                        inputs[:, drop] = 0.0
                    return super().forward(inputs, *a, **kw)

            m = Masked(channels=cfg.channels, hidden_mult=cfg.hidden_mult,
                       n_sub_models=cfg.n_sub_models)
            m, _ = train(cfg, splits["train"], splits["val"], model=m,
                         log=lambda s: None)
            _, vd = evaluate(m, splits["val"], cfg.infer_steps)
            scores.append(round(vd, 4))
        raw[name] = scores
        mean = sum(scores) / len(scores)
        spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
        res[name] = {"val_dice": round(mean, 4), "seeds": scores,
                     "spread": round(spread, 4)}
        base = (res.get("all channels") or {}).get("val_dice")
        # A delta smaller than the seed spread of the baseline is not a finding.
        delta = ""
        if base is not None and drop is not None:
            d = mean - base
            floor = res["all channels"].get("spread", 0.0)
            verdict = "within seed noise" if abs(d) <= floor else "beyond seed noise"
            delta = f"   ΔDSC {d:+.4f}  ({verdict}, noise floor {floor:.4f})"
        print(f"[ablation] {name:30s} val DSC {mean:.4f} "
              f"{scores if seeds > 1 else ''}{delta}", flush=True)
        (out / "channel_ablation.json").write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--data", type=Path, default=ROOT / "data_liver")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=120,
                    help="per arm; the studies are comparative, not the shipped model")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeats per arm; >1 gives the noise floor a delta "
                         "must clear before it is a finding")
    ap.add_argument("--arch", choices=("fast", "full"), default="fast",
                    help="architecture for the comparative studies; the CNN "
                         "baseline always matches the reference model")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    splits = load_splits(a.data, device=a.device)
    print(f"[data] " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))

    if a.baseline or a.all:
        study_baseline(splits, a.device, a.out, a.epochs)
    if a.stability or a.all:
        study_stability(splits, a.device, a.out, a.epochs, arch=a.arch)
    if a.ablation or a.all:
        study_channel_ablation(splits, a.device, a.out, a.epochs, arch=a.arch,
                               seeds=a.seeds)


if __name__ == "__main__":
    main()
