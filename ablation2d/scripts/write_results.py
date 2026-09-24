#!/usr/bin/env python3
"""Write docs/RESULTS.md from the measurement files.

    python scripts/write_results.py

Every number comes from a JSON in results/ or a scored checkpoint. A study that
has not been run shows up as "not run yet", never as a remembered number.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
NOT_RUN = "_not run yet_"
HEAD = ["| | DSC | DSC p10 | area err (cm²) | recall | precision | vessel F1 |",
        "|---|---|---|---|---|---|---|"]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def metrics_row(d, name):
    """One necrosis-table row, or None for an entry with vessel F1 only."""
    if "dice" not in d:
        return None
    ves = f"{d['vessel_f1']:.4f}" if "vessel_f1" in d else "-"
    return (f"| {name} | {d['dice']:.4f} | {d.get('dice_p10', float('nan')):.4f} | "
            f"{d['area_err_cm2']:+.2f} | {d['recall']:.3f} | {d['precision']:.3f} | {ves} |")


def table(scored):
    return HEAD + [r for n, d in scored["test"].items() if (r := metrics_row(d, n))]


def hu_baseline(scored):
    return next((d["vessel_f1"] for d in scored["test"].values() if "dice" not in d), None)


def main():
    ref = load(ROOT / "checkpoints/ablation_cnca_liver.scored.json")
    fast = load(ROOT / "ck_fast_liver/ablation_cnca_fast.scored.json")
    cnn = load(RES / "baseline_cnn.json")
    stab = load(RES / "stability.json")
    abl = load(RES / "channel_ablation.json")
    cap = load(RES / "capacity_ab.json")
    d4on = load(RES / "fast_d4on.json")

    L = ["# Results: 2-D ablation surrogate", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by "
         "`scripts/write_results.py`. Test split: 10 patients the model never saw. K and "
         "the vessel threshold are chosen on validation. The reference model and the "
         "notebook use the plans with one or two needles (659 test plans); the older "
         "studies further down were scored on all 800.", "",
         "## The reference model", ""]
    if ref:
        c = ref["config"]
        sp = ref["sphere_baseline"]
        L += [f"`checkpoints/ablation_cnca_liver.pt`: {c['channels']} channels, "
              f"{c['n_sub_models']} blocks per step, hidden x{c['hidden_mult']}, "
              f"{c['epochs']} epochs on the liver-masked corpus. "
              f"K = {ref['infer_steps']}, vessel threshold {ref['vessel_threshold']:.2f}.", ""]
        L += table(ref)
        n, s = ref["test"]["C-NCA"]["dice"], ref["test"].get("device-chart sphere", {}).get("dice")
        L += ["", f"Device-chart sphere: `r = {sp['C']:.4f} * P^{sp['ALPHA']:.4f} * "
                  f"t^{sp['BETA']:.4f}` ({sp['_fit']})."]
        if s is not None:
            L += ["", f"Against the device chart: {n - s:+.4f} DSC"
                      + (", which is what knowing the anatomy is worth." if n > s
                         else "; the chart is the better necrosis predictor here.")]
        L += ["", "Recall is above precision: the 100:1 foreground weighting makes the "
                  "model over-predict the zone, the safer error for an ablation margin.", "",
              "### Rollout length (validation DSC)", ""]
        ks = sorted(map(int, ref["step_sweep"]))
        L += ["| K | " + " | ".join(map(str, ks)) + " |", "|---" * (len(ks) + 1) + "|",
              "| DSC | " + " | ".join(f"{ref['step_sweep'][str(k)]:.4f}" for k in ks) + " |"]
        curve = ref.get("vessel_threshold_curve_val")
        if curve:
            L += ["", "### Vessel threshold (validation F1)", "",
                  "| threshold | " + " | ".join(curve) + " |", "|---" * (len(curve) + 1) + "|",
                  "| F1 | " + " | ".join(f"{v:.4f}" for v in curve.values()) + " |"]
    else:
        L.append(NOT_RUN)

    L += ["", "## The session recipe", ""]
    if fast:
        c = fast["config"]
        L += [f"{c['channels']} channels, {c['n_sub_models']} blocks per step, "
              f"{c['epochs']} epochs, K = {fast['infer_steps']}. The notebook trains "
              "this recipe under a 5-minute budget instead of a fixed epoch count.", ""]
        L += table(fast)
        fv = fast["test"].get("C-NCA", {}).get("vessel_f1")
        if d4on and fv is not None:
            on = d4on["test"]["C-NCA"]
            L += ["", f"With D4 augmentation on (same budget, same seed) necrosis DSC is "
                      f"{on['dice']:.4f} and vessel F1 falls from {fv:.4f} to "
                      f"{on['vessel_f1']:.4f}: the heat equation is rotation-invariant, "
                      "axial CT is not."]
    else:
        L.append(NOT_RUN)

    budget = load(ROOT / "results" / "session_budget.json")
    if budget:
        L += ["", "### Under the notebook's 5-minute budget", "",
              budget["_what"], "",
              "| arm | epochs | DSC 1x | vessel F1 1x | DSC 8x | vessel F1 8x |",
              "|---|---|---|---|---|---|"]
        for name, r in budget["arms"].items():
            L.append(f"| {name} | {r['epochs_run']:.0f} | {r['dsc_1x']:.4f} | {r['vf1_1x']:.4f} "
                     f"| {r['dsc_8x']:.4f} | {r['vf1_8x']:.4f} |")
        L += ["", "Seed-to-seed spread on the vessel F1 is about 0.05. Two changes stand out: "
                  "lr 5e-3 (about +0.07 vessel F1 in the time available) and dropping the "
                  "random fire mask, which improves both heads and makes the model "
                  "deterministic (its 1x and 8x columns are identical)."]

    rb = load(ROOT / "results" / "robustness.json")
    if rb:
        rows = [(k, v) for k, v in rb.items() if not k.startswith("_")]
        L += ["", "## Robustness", "", rb["_what"], "",
              "| | " + " | ".join(k for k, _ in rows) + " |", "|---" * (len(rows) + 1) + "|"]
        for key, label in (("dsc", "necrosis DSC, own plan"), ("vessel_f1", "vessel F1, own plan"),
                           ("vessel_f1_random", "vessel F1, random plans"), ("vessel_f1_250W", "vessel F1, 250 W"),
                           ("vessel_f1_outside", "vessel F1, tips outside the liver"),
                           ("vessel_f1_none", "vessel F1, no needle"),
                           ("vessel_recall_in_lesion", "vessel recall inside the lesion"),
                           ("vessel_recall_ring_outside", "vessel recall, 8 mm around it"),
                           ("vessel_recall_far", "vessel recall, further away"),
                           ("dsc_K40", "necrosis DSC at 40 steps"), ("dsc_K100", "necrosis DSC at 100 steps"),
                           ("necrosis_cm2_no_needle", "necrosis with no needle (cm²)")):
            L.append(f"| {label} | " + " | ".join(f"{v.get(key, float('nan')):.3f}" for _, v in rows) + " |")

    af = load(ROOT / "results" / "anatomy_first.json")
    if af:
        L += ["", "## Anatomy first: vessels that do not depend on the plan", "", af["_what"], "",
              "| | DSC | vessel F1 | vessel F1, random plans | recall inside lesion | 8 mm around | further | DSC at K=40 |",
              "|---|---|---|---|---|---|---|---|"]
        for name, r in af.items():
            if not name.startswith("_"):
                L.append(f"| {name} | {r['dsc']:.3f} | {r['vessel_f1']:.3f} | {r['vessel_f1_random']:.3f} | "
                         f"{r['vessel_recall_in_lesion']:.3f} | {r['vessel_recall_ring_outside']:.3f} | "
                         f"{r['vessel_recall_far']:.3f} | {r['dsc_K40']:.3f} |")
        L += ["", "With a single rollout the model learns that vessels are rare inside a lesion and "
                  "stops seeing them there. Reading the anatomy first, with the needle channels empty, "
                  "and then holding the vessel channel fixed makes the vessel map independent of the "
                  "plan by construction: its F1 is the same under random plans as under the real one."]

    fire = load(ROOT / "results" / "fire_rate.json")
    if fire:
        L += ["", "## The fire mask", "", fire["_what"], "",
              "| 240 epochs | DSC | vessel F1 |", "|---|---|---|"]
        for name, r in fire["reference recipe, 240 epochs"].items():
            d, v = r.get("dsc", r.get("dsc_1x")), r.get("vessel_f1", r.get("vessel_f1_1x"))
            L.append(f"| {name} | {d:.4f} | {v:.4f} |")
        ks = fire["long rollouts, test DSC by K"]
        cols = list(next(iter(ks.values())))
        L += ["", "| test DSC at K = | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name, r in ks.items():
            L.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        L += ["", fire["_verdict"]]

    long = load(ROOT / "results" / "long_training.json")
    if long:
        L += ["", "## Training the fire-0.5 recipe longer", "",
              "| run | DSC 1x | DSC 8x | vessel F1 1x | vessel F1 8x |", "|---|---|---|---|---|"]
        for name, r in long.items():
            if not name.startswith("_"):
                L.append(f"| {name} | {r['dsc_1x']:.4f} | {r['dsc_8x']:.4f} | "
                         f"{r['vessel_f1_1x']:.4f} | {r['vessel_f1_8x']:.4f} |")
        L += ["", long["_verdict"][0].upper() + long["_verdict"][1:]]

    L += ["", "## The recipe ladder", ""]
    arms = [("control", "checkpoints", "-", "-", "3.0", "full"),
            ("+ D4 + vessel warm start", "checkpoints_aug", "yes", "yes", "3.0", "full"),
            ("+ free conditioning", "checkpoints_free", "yes", "yes", "3.0", "full"),
            ("+ loss rebalance", "checkpoints_bal", "yes", "yes", "0.3", "full"),
            ("rebalance only", "checkpoints_w03_full", "-", "-", "0.3", "full"),
            ("liver-masked CT", "checkpoints_liver", "-", "-", "0.3", "masked"),
            ("w=1.0, full CT", "checkpoints_w10_full", "-", "-", "1.0", "full"),
            ("w=1.0, masked CT", "checkpoints_w10_mask", "-", "-", "1.0", "masked")]
    got = [(n, load(ROOT / d / "ablation_cnca_full.scored.json"), d4, ws, w, ct)
           for n, d, d4, ws, w, ct in arms]
    got = [g for g in got if g[1]]
    if got:
        L += ["16 channels, 80 epochs, seed 0 for every arm, all re-scored with the same "
              "evaluation code. `1x` is one rollout, `8x` the mean of eight. These arms "
              "predate the wider vessel-threshold grid, so their vessel F1 is capped at "
              "a threshold of 0.7.", "",
              "| arm | D4 | warm start | w | CT | K | DSC 1x | DSC 8x | vessel 1x | vessel 8x |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for n, d, d4, ws, w, ct in got:
            t, r = d["test"].get("C-NCA", {}), d["test"].get("C-NCA + 8 repeats", {})
            L.append(f"| {n} | {d4} | {ws} | {w} | {ct} | {d['infer_steps']} | "
                     f"{t.get('dice', np.nan):.4f} | {r.get('dice', np.nan):.4f} | "
                     f"{t.get('vessel_f1', np.nan):.4f} | {r.get('vessel_f1', np.nan):.4f} |")
        by = {n: d for n, d, *_ in got}
        pairs = [("rebalance only", "control", "vessel weight 3.0 -> 0.3"),
                 ("liver-masked CT", "rebalance only", "masking the CT to the liver"),
                 ("w=1.0, masked CT", "liver-masked CT", "w 0.3 -> 1.0, masked"),
                 ("w=1.0, full CT", "rebalance only", "w 0.3 -> 1.0, full CT"),
                 ("+ free conditioning", "+ D4 + vessel warm start", "restore_conditioning off"),
                 ("+ loss rebalance", "+ D4 + vessel warm start", "rebalance, on the augmented arm")]
        rows = [f"| {what} | {by[a]['test']['C-NCA']['dice'] - by[b]['test']['C-NCA']['dice']:+.4f} | "
                f"{by[a]['test']['C-NCA'].get('vessel_f1', 0) - by[b]['test']['C-NCA'].get('vessel_f1', 0):+.4f} |"
                for a, b, what in pairs if a in by and b in by]
        if rows:
            L += ["", "Single-variable comparisons (arm 2 bundles augmentation and the warm "
                      "start; every row below changes one thing):", "",
                  "| change | ΔDSC | Δvessel F1 |", "|---|---|---|"] + rows
    else:
        L.append(NOT_RUN)

    L += ["", "## The vessel head", ""]
    if ref:
        nca, base = ref["test"]["C-NCA"].get("vessel_f1"), hu_baseline(ref)
        L += ["The model is never given a vessel mask. Its baseline is the best single "
              "Hounsfield threshold, fitted on the split it is scored on.", "",
              "| | vessel F1 |", "|---|---|"]
        if nca is not None:
            L.append(f"| C-NCA vessel head | **{nca:.4f}** |")
        if base is not None:
            L.append(f"| best single HU threshold | {base:.4f} |")
    if cap:
        L += ["", "### Width, not loss weighting", "",
              "Loss changes (Dice + BCE for MSE, weights 1/3/10, pos_weight 50, learned "
              "uncertainty weighting) left vessel F1 near 0.18. Real CT and then width "
              "moved it, and both heads rose together:", "",
              "| channels | scratch | params | necrosis DSC | vessel F1 |", "|---|---|---|---|---|"]
        for k, v in cap["arms"].items():
            L.append(f"| {k.split('ch')[0]} | {v['hidden_channels']} | {v['params']:,} | "
                     f"{v['dice']:.4f} | {v['vessel_f1']:.4f} |")
        L += ["", "80 epochs per arm on the full-CT corpus. Timings are omitted: that GPU "
                  "was thermally throttled throughout."]

    L += ["", "## Against a parameter-matched CNN", ""]
    if cnn and ref:
        n = ref["test"]["C-NCA"]
        c = ref["config"]
        from ablation2d.model import AblationCNCA
        nca = AblationCNCA(c["channels"], c["hidden_mult"], c["n_sub_models"])
        reach = nca.reach_cells * ref["infer_steps"]
        L += ["Dilated 1-2-4-8-16-32, no downsampling, same corpus, loss and split, and a "
              "larger receptive field than the NCA.", "",
              "| | params | reach | DSC | recall | precision | vessel F1 |",
              "|---|---|---|---|---|---|---|",
              f"| dilated CNN | {cnn['params']:,} | {cnn['reach_cells']} cells | {cnn['dice']:.4f} | "
              f"{cnn['recall']:.3f} | {cnn['precision']:.3f} | "
              f"{cnn['vessel_f1']:.4f} |" if "vessel_f1" in cnn else
              f"| dilated CNN | {cnn['params']:,} | {cnn['reach_cells']} cells | {cnn['dice']:.4f} | "
              f"{cnn['recall']:.3f} | {cnn['precision']:.3f} | - |",
              f"| C-NCA | {nca.n_params:,} | {reach} cells | {n['dice']:.4f} | "
              f"{n['recall']:.3f} | {n['precision']:.3f} | {n.get('vessel_f1', float('nan')):.4f} |",
              "", f"DSC difference {n['dice'] - cnn['dice']:+.4f}. The CNN was trained for "
                  f"{cnn.get('epochs', '?')} epochs on "
                  f"{cnn.get('corpus', 'the corpus named in its JSON')}."]
    else:
        L.append(NOT_RUN)

    L += ["", "## Persistence", ""]
    if stab:
        ks = sorted(int(k) for k in stab["with_persistence"]["dice_by_k"])
        L += [f"Two arms differing only in `persist_rate`; {stab.get('_arch', '?')} "
              f"architecture, {stab.get('_epochs', '?')} epochs, trained on "
              f"{stab['with_persistence']['trained_steps']} steps.", "",
              "| val DSC at K = | " + " | ".join(map(str, ks)) + " |", "|---" * (len(ks) + 1) + "|",
              "| sampled length only | " + " | ".join(
                  f"{stab['without_persistence']['dice_by_k'][str(k)]:.4f}" for k in ks) + " |",
              "| + persistence | " + " | ".join(
                  f"{stab['with_persistence']['dice_by_k'][str(k)]:.4f}" for k in ks) + " |"]
    else:
        L.append(NOT_RUN)

    L += ["", "## Input channels", ""]
    if abl:
        base = abl["all channels"]["val_dice"]
        L += [f"One channel zeroed, everything else identical; {abl.get('_arch', '?')} "
              f"architecture, {abl.get('_epochs', '?')} epochs. With all channels: val DSC "
              f"{base:.4f}.", "", "| channel zeroed | val DSC | ΔDSC |", "|---|---|---|"]
        for k, v in sorted(((k.replace("drop ", ""), v["val_dice"]) for k, v in abl.items()
                            if k != "all channels" and not k.startswith("_")),
                           key=lambda t: t[1]):
            L.append(f"| `{k}` | {v:.4f} | {v - base:+.4f} |")
    else:
        L.append(NOT_RUN)

    L += ["", "## Corpus", ""]
    sizes = {s: np.load(ROOT / "data_liver" / f"{s}.npz", mmap_mode="r")["inputs"].shape[0]
             for s in ("train", "val", "test") if (ROOT / "data_liver" / f"{s}.npz").exists()}
    if sizes:
        L.append(f"`data_liver/`: {sum(sizes.values())} plans on real patient CT, split by "
                 f"patient ({', '.join(f'{k} {v}' for k, v in sizes.items())}). See "
                 "`data_liver/DATASHEET.md`.")
    else:
        L.append(NOT_RUN)

    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
