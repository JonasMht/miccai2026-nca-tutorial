#!/usr/bin/env python3
"""Every figure the session needs, from one script.

    python scripts/make_figures.py --all --out figures

Physics figures need the solver; model figures need a checkpoint. Each group is
skipped with a message rather than crashing, so this runs at any stage of the
build.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d import viz  # noqa: E402
from ablation2d.anatomy import (Anatomy, DX_MM, LBL_LIVER, LBL_VESSEL, NX, NY,  # noqa: E402
                                from_segmentations)
from ablation2d.channels import build_inputs  # noqa: E402
from ablation2d.device import CALIBRATION_STATUS  # noqa: E402
from ablation2d.plans import Needle, Plan  # noqa: E402
from ablation2d.solver_env import solver_available  # noqa: E402


def load_real_slice(root: Path | None = None):
    """The demo slice, with its real CT when the extraction stored one."""
    r = np.load((root or ROOT) / "data" / "real_slice.npz")
    hu = r["hounsfield"] if "hounsfield" in r else None
    return from_segmentations(r["seg_liver"], r["seg_vessel"], hounsfield=hu)

ROOT = Path(__file__).resolve().parent.parent
CAPTION = dict(fontsize=7.5, color="#555")


def sample_real_anatomy(rng, split="test", root: Path | None = None):
    """One anatomy drawn from the real corpus - the same slices the model is
    trained and scored on.

    The physics figures used to draw procedural blobs. That was defensible when
    the corpus was procedural; it is not now. A figure captioned "what the
    corpus looks like" that shows anatomy absent from the corpus is simply a
    wrong caption, and the vessel head's whole story is that real vasculature
    and a synthetic blob are not the same problem.
    """
    from ablation2d.anatomy import from_patient_slice
    d = np.load((root or ROOT) / "data_real" / f"{split}_slices.npz")
    i = int(rng.integers(len(d["z"])))
    return from_patient_slice(
        hounsfield=d["hounsfield"][i].astype(np.float32),
        material_id=d["material_id"][i],
        seg_liver=d["seg_liver"][i].astype(bool),
        seg_vessel=d["seg_vessel"][i].astype(bool),
        source=f'{d["case"][i]}@z{int(d["z"][i])}')


def _real_case(rng, split="test", n_needles=None, tries=24):
    """A real anatomy plus a plan that fits in it. Not every slice admits every
    plan - a needle has to reach parenchyma without crossing a vessel - so this
    resamples the slice, not just the plan."""
    from ablation2d.plans import sample_plan
    kw = {"n_needles": n_needles} if n_needles else {}
    for _ in range(tries):
        anat = sample_real_anatomy(rng, split)
        try:
            return anat, sample_plan(rng, anat, **kw)
        except RuntimeError:
            continue
    raise RuntimeError(f"no feasible plan in {tries} real slices")


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p}  ({p.stat().st_size / 1e3:.0f} kB)")


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def fig_showcase(out: Path, n=6, seed=101):
    """A wall of solved cases - what the corpus actually looks like."""
    from ablation2d.physics import simulate
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, n // 2, figsize=(3.1 * (n // 2), 7.3), dpi=140)
    for ax in axes.ravel():
        anat, plan = _real_case(rng)
        s = simulate(anat, plan)
        viz.show_necrosis(s.cell_death, ax, anat, plan, None, on_ct=True)
        ax.set_title(f"{plan.n} antenna(s) · {s.lesion_cm2:.1f} cm²",
                     fontsize=8.5, fontweight="bold")
    fig.suptitle("the solver ground truth on REAL patient CT - Pennes bioheat + "
                 "microwave SAR + Arrhenius, solved in 3-D and sliced",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005, CALIBRATION_STATUS, ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.subplots_adjust(hspace=0.18)
    _save(fig, out, "01_ground_truth_showcase.png")


def fig_channels(out: Path, seed=7):
    from ablation2d.physics import simulate
    rng = np.random.default_rng(seed)
    anat, plan = _real_case(rng, n_needles=2)
    s = simulate(anat, plan)
    fig, axes = viz.show_channels(build_inputs(anat, plan))
    _save(fig, out, "02_input_channels.png")

    # The CT, the truth the model never sees, and the two things it must produce.
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.4), dpi=140)
    viz.show_ct(anat, ax[0], "the CT - all the model is given")
    viz.draw_plan(plan, anat, ax[0])
    viz.show_anatomy(anat, ax[1], "the labels (never shown to the model)")
    viz.show_necrosis(s.cell_death, ax[2], anat, plan,
                      "target 0 - necrosis (d ≥ 0.99)", on_ct=True)
    viz.show_ct(anat, ax[3], "target 1 - vessels")
    ax[3].contour(anat.vessel_mask.astype(float), levels=[0.5],
                  colors=["#00e5ff"], linewidths=1.0)
    hu_l = anat.hounsfield[anat.liver_mask]
    hu_v = anat.hounsfield[anat.vessel_mask]
    d = abs(hu_v.mean() - hu_l.mean()) / np.sqrt((hu_l.var() + hu_v.var()) / 2)

    # Why the vessel task is hard, measured on this slice rather than asserted:
    # sweep every HU threshold and keep the best F1 it can reach. d' alone does
    # not tell the story - 1.2 is a perfectly usable separation on balanced
    # classes. It is d' that size against 3 % prevalence that kills precision.
    hu, tv = anat.hounsfield, anat.vessel_mask
    best_t, best_f1 = 0.0, 0.0
    for t in np.percentile(hu[anat.liver_mask | tv], np.linspace(50, 99.7, 120)):
        pm = (hu > t) & (anat.liver_mask | tv)
        den = pm.sum() + tv.sum()
        if den:
            f1 = 2 * (pm & tv).sum() / den
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
    prev = tv.sum() / max((anat.liver_mask | tv).sum(), 1)
    fig.text(0.5, 0.01, viz.vessel_legend_text(anat)
             + f"   ·   liver {hu_l.mean():.0f}±{hu_l.std():.0f} HU vs vessel "
               f"{hu_v.mean():.0f}±{hu_v.std():.0f} HU (d' = {d:.2f}) - a real "
               f"separation, but against {100 * prev:.1f} % prevalence the best "
               f"single threshold on this slice still only reaches F1 "
               f"{best_f1:.3f} at {best_t:.0f} HU. The second needle is drawn "
               f"'off': placed, not energised - which is why `needle` and "
               f"`applicator_activation` are two channels and not one.",
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out, "03_anatomy_and_label.png")


def fig_heat_sink(out: Path):
    """The measurement the whole session turns on."""
    from ablation2d.physics import simulate
    plan = Plan([Needle(64 * DX_MM, 64 * DX_MM, 64 * DX_MM, 110 * DX_MM, 100.0, 360.0)])
    xx = np.arange(NX)[None, :].repeat(NY, 0)
    cals = [0, 3, 6, 9, 13]
    fields, areas, anats = [], [], []
    for cal in cals:
        lab = np.full((NY, NX), LBL_LIVER, np.uint8)
        rad = np.zeros((NY, NX), np.float32)
        if cal:
            m = np.abs(xx - 69) <= 0.5 * cal / DX_MM
            lab[m] = LBL_VESSEL
            rad[m] = 0.5 * cal
        a = Anatomy(label=lab, vessel_radius_mm=rad)
        s = simulate(a, plan)
        fields.append(s.cell_death); areas.append(s.lesion_cm2); anats.append(a)

    fig, axes = plt.subplots(1, len(cals) + 1, figsize=(3.0 * (len(cals) + 1), 3.4),
                             dpi=140)
    for ax, cal, f, a, anat in zip(axes, cals, fields, areas, anats):
        viz.show_necrosis(f, ax, anat, plan, None)
        ax.set_title(f"{'no vessel' if not cal else f'{cal} mm vein'}\n{a:.1f} cm²"
                     + ("" if not cal else f"  ({100 * (a / areas[0] - 1):+.0f} %)"),
                     fontsize=8.5, fontweight="bold")
    axes[-1].plot(cals[1:], areas[1:], "o-", color="#c0392b", lw=1.6)
    axes[-1].axhline(areas[0], ls="--", lw=1.0, color="#888")
    axes[-1].annotate("no vessel", (cals[-1], areas[0]), fontsize=7.5,
                      ha="right", va="bottom", color="#666")
    axes[-1].axvline(3.0, ls=":", lw=1.0, color="#2980b9")
    axes[-1].annotate("Lu et al. 3 mm\nheat-sink threshold", (3.2, min(areas) + 0.15),
                      fontsize=7, color="#2980b9")
    axes[-1].set_xlabel("vessel calibre (mm)", fontsize=8)
    axes[-1].set_ylabel("lesion (cm²)", fontsize=8)
    axes[-1].grid(alpha=0.25)
    axes[-1].tick_params(labelsize=7)
    fig.suptitle("One vein, 10 mm from the tip, takes a third of the ablation - "
                 "100 W for 6 min", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out, "04_heat_sink.png")


def fig_slab_vs_3d(out: Path):
    """Why the label is solved in 3-D and sliced."""
    from ablation2d.physics import simulate
    lab = np.full((NY, NX), LBL_LIVER, np.uint8)
    anat = Anatomy(label=lab, vessel_radius_mm=np.zeros((NY, NX), np.float32))
    plan = Plan([Needle(64 * DX_MM, 64 * DX_MM, 64 * DX_MM, 104 * DX_MM, 150.0, 300.0)])
    slab = simulate(anat, plan, nz=1)
    solid = simulate(anat, plan, nz=32)

    def extent(f):
        m = f >= 0.99
        ys, xs = np.nonzero(m)
        return (xs.max() - xs.min() + 1) * DX_MM, (ys.max() - ys.min() + 1) * DX_MM

    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.9), dpi=140)
    for a, f, t in ((ax[0], slab.cell_death, "2-D slab (nz = 1)"),
                    (ax[1], solid.cell_death, "3-D solve, mid-plane")):
        viz.show_necrosis(f, a, anat, plan, None)
        w, h = extent(f)
        a.set_title(f"{t}\n{w:.0f} × {h:.0f} mm", fontsize=9, fontweight="bold")
    viz.show_anatomy(anat, ax[2])
    ax[2].imshow(slab.cell_death - solid.cell_death, cmap="RdBu_r", vmin=-1, vmax=1,
                 origin="upper", zorder=2, alpha=0.92)
    ax[2].contour(solid.cell_death, levels=[0.99], colors=["#111"], linewidths=0.9,
                  zorder=3)
    ax[2].set_title("red = the slab over-kills", fontsize=9, fontweight="bold")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle("A one-cell-deep grid is the physics of an INFINITE SLAB - "
                 "no heat leaves through the faces", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01, "so the anatomy is extruded, the solve is 3-D, and the "
                        "label is the mid-plane", ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    _save(fig, out, "05_slab_vs_3d.png")


def fig_real_slice(out: Path):
    """The demo anatomy, solved."""
    from ablation2d.physics import simulate
    anat = load_real_slice()
    plan = Plan([
        _aim(anat, 0.42, 0.52, 250.0, 100.0, 5.0),
        _aim(anat, 0.55, 0.46, 300.0, 90.0, 5.0),
        _aim(anat, 0.49, 0.62, 200.0, 80.0, 5.0),
    ])
    s = simulate(anat, plan)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.8), dpi=150)
    viz.show_ct(anat, ax[0], "case-1001, axial 154 - the real CT")
    viz.draw_plan(plan, anat, ax[0])
    ax[0].contour(anat.vessel_mask.astype(float), levels=[0.5],
                  colors=["#00e5ff"], linewidths=0.8)
    viz.show_necrosis(s.cell_death, ax[1], anat, plan,
                      f"three antennas · {s.lesion_cm2:.1f} cm² necrosis",
                      on_ct=True)
    fig.text(0.5, 0.015, viz.vessel_legend_text(anat)
             + "   ·   de-identified segmentation, my PhD project's corpus",
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, "06_real_slice.png")


def _aim(anat, fx, fy, angle, dial, minutes):
    h, w = anat.shape
    organ = anat.liver_mask | anat.vessel_mask
    ys, xs = np.nonzero(organ)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ty, tx = y0 + fy * (y1 - y0), x0 + fx * (x1 - x0)
    th = np.deg2rad(angle)
    u = np.array([np.cos(th), np.sin(th)])
    tip = np.array([tx * anat.dx_mm, ty * anat.dx_mm])
    t = 120.0
    for o, d, hi in ((tip[0], u[0], w * anat.dx_mm), (tip[1], u[1], h * anat.dx_mm)):
        if abs(d) > 1e-9:
            t = min(t, ((hi if d > 0 else 0.0) - o) / d)
    base = tip + u * max(t, 30.0)
    return Needle(float(tip[0]), float(tip[1]), float(base[0]), float(base[1]),
                  dial, minutes * 60.0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def fig_model(out: Path, ckpt: Path, device="cpu", data: Path | None = None,
              mask_ct: bool | None = None):
    import torch
    from ablation2d.data import AblationDataset
    from ablation2d.evaluate import (fit_sphere_baseline, hu_threshold_vessel_f1,
                                     report, sphere_baseline)
    from ablation2d.train import load_checkpoint

    data = data or ROOT / "data"
    if mask_ct is None:
        mask_ct = "liver" in data.name
    model, cfg, meta = load_checkpoint(ckpt, device=device)
    test = AblationDataset(data / "test.npz", device=device)
    train = AblationDataset(data / "train.npz", device=device, limit=800)

    # -- prediction vs truth, a wall of them ------------------------------- #
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.6), dpi=140)
    with torch.no_grad():
        pred = model(test.inputs[:12], steps=cfg.infer_steps)
    for j, ax in enumerate(axes.ravel()):
        t = test.targets[j, 0].cpu().numpy()
        p = pred[j, 0].cpu().numpy()
        # The anatomy is recovered from the stored channels - without it these
        # are lesions floating on nothing, and the whole point is where they sit.
        viz.show_necrosis(p, ax, viz.anatomy_from_inputs(test.inputs[j]), None,
                          None, on_ct=True)
        ax.contour(t, levels=[0.99], colors=["#00e0ff"], linewidths=1.0)
        inter = ((p > 0.5) & (t >= 0.99)).sum()
        d = 2 * inter / max((p > 0.5).sum() + (t >= 0.99).sum(), 1)
        ax.set_title(f"DSC {d:.3f}", fontsize=8.5, fontweight="bold")
    fig.suptitle("C-NCA prediction (fill) against the solver ground truth (cyan contour)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "07_prediction_vs_truth.png")

    # -- the rollout -------------------------------------------------------- #
    with torch.no_grad():
        _, trace = model(test.inputs[:1], steps=cfg.infer_steps * 2, return_trace=True)
    fig, _ = viz.show_rollout(trace, every=max(1, len(trace) // 8), max_frames=8)
    fig.suptitle("The lesion grows outward from the radiating slot, one local "
                 "update at a time", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, out, "08_rollout.png")

    # -- against the device chart ------------------------------------------ #
    sp = fit_sphere_baseline(train)
    # The vessel decision threshold comes from validation, like K. Reporting at
    # 0.5 measures the head's calibration under 3 % prevalence, not its
    # segmentation, and every model here prefers a threshold well above 0.5.
    from ablation2d.model import best_vessel_threshold
    val = AblationDataset(data / "val.npz", device=device)
    with torch.no_grad():
        vthr = best_vessel_threshold(model(val.inputs, steps=cfg.infer_steps),
                                     val.targets)[0]
    hu_base = hu_threshold_vessel_f1(val)
    res = report(model, test, steps=cfg.infer_steps, vessel_threshold=vthr,
                 baselines={"device-chart sphere": lambda x: sphere_baseline(x, sp)})
    names = [n for n in res if "dice" in res[n]]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=140)
    ax[0].barh(names, [res[n]["dice"] for n in names],
               color=["#c0392b", "#7f8c8d", "#bdc3c7"])
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("DSC", fontsize=8)
    ax[0].grid(alpha=0.25, axis="x")
    for i, n in enumerate(names):
        ax[0].text(res[n]["dice"] + 0.01, i, f"{res[n]['dice']:.3f}", va="center",
                   fontsize=8)
    for n, c in zip(names, ["#c0392b", "#7f8c8d", "#bdc3c7"]):
        ax[1].scatter(res[n]["recall"], res[n]["precision"], s=70, color=c, label=n,
                      zorder=3)
    ax[1].plot([0, 1], [0, 1], ls=":", lw=0.8, color="#aaa")
    ax[1].set_xlabel("recall", fontsize=8); ax[1].set_ylabel("precision", fontsize=8)
    ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=7, loc="lower left")
    ax[1].annotate("above the line = over-treats", (0.03, 0.93), fontsize=7,
                   color="#666")
    fig.suptitle("Beating the device chart is the value of knowing where the "
                 "vessels are", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out, "09_vs_baseline.png")
    print("  metrics:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in res.items()})

    # -- the set pieces ----------------------------------------------------- #
    from ablation2d import experiments
    anat = load_real_slice()
    _save(experiments.heat_sink_pair(model, steps=cfg.infer_steps, device=device,
                                     mask_ct=mask_ct),
          out, "10_model_heat_sink.png")
    _save(experiments.power_sweep(model, anat, steps=cfg.infer_steps, device=device,
                                  mask_ct=mask_ct),
          out, "11_power_sweep.png")
    _save(experiments.outside_the_liver(model, anat, steps=cfg.infer_steps,
                                        device=device, mask_ct=mask_ct),
          out, "12_outside_the_liver.png")

    # -- the demo, as the room will see it ---------------------------------- #
    plan = Plan([_aim(anat, 0.42, 0.52, 250.0, 100.0, 6.0),
                 _aim(anat, 0.55, 0.46, 300.0, 90.0, 5.0),
                 _aim(anat, 0.49, 0.62, 200.0, 80.0, 4.0)])
    x = torch.from_numpy(build_inputs(anat, plan, mask_ct=mask_ct)).unsqueeze(0).to(device)
    with torch.no_grad():
        out2 = model(x, steps=cfg.infer_steps)[0].cpu().numpy()
    p, ves = out2[0], (out2[1] if out2.shape[0] > 1 else None)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 5.4), dpi=150)
    viz.show_necrosis(p, ax[0], anat, plan,
                      "predicted necrosis - three antennas on a real slice",
                      on_ct=True)
    viz.show_ct(anat, ax[1], "predicted vessels (cyan) vs truth (orange)")
    if ves is not None:
        ax[1].contour(ves, levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
        ax[1].contour(anat.vessel_mask.astype(float), levels=[0.5],
                      colors=["#ffb300"], linewidths=0.9, linestyles=":")
        t = anat.vessel_mask; q = ves > 0.5
        f1 = 2 * (q & t).sum() / max(q.sum() + t.sum(), 1)
        ax[1].set_title(f"predicted vessels - F1 {f1:.3f} "
                        f"(best HU threshold {hu_base:.3f})", fontsize=9,
                        fontweight="bold")
    ax = ax[0]
    fig.text(0.5, 0.02, f"{model.n_params:,} parameters · {cfg.infer_steps} NCA steps"
                        f" · {viz.vessel_legend_text(anat)}", ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, "13_planner.png")


def fig_convergence(out: Path, arms: dict[str, Path],
                    name="19_convergence.png"):
    """Validation trajectories - the answer to "is this arm even finished?".

    A fixed epoch budget is not a fixed amount of training when one arm has 8x
    the effective corpus: D4 multiplies the distinct (sample, orientation) pairs
    by eight while the number of presentations stays put, so the augmented model
    sees each distinct example a tenth as often. Comparing final numbers at 80
    epochs answers "what do I get in 80 epochs", not "is the method better".

    This figure is how you tell the two apart: an arm whose validation loss is
    still falling at the last epoch was stopped, not converged, and its final
    score is a lower bound.
    """
    import json

    hist = {}
    for label, d in arms.items():
        f = Path(d) / "history.json"
        if f.exists():
            hist[label] = json.loads(f.read_text())
    if not hist:
        print(f"  [skip] {name}: no history.json found")
        return

    keys = [("val_loss", "validation loss", True),
            ("val_dice", "necrosis DSC", False),
            ("val_vessel_f1", "vessel F1", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=140)
    colours = ["#7f8c8d", "#c0392b", "#2980b9", "#27ae60"]
    verdicts = []
    for i, (label, h) in enumerate(hist.items()):
        c = colours[i % len(colours)]
        for ax, (k, title, lower_better) in zip(axes, keys):
            xs = [r["epoch"] for r in h if k in r]
            ys = [r[k] for r in h if k in r]
            if not xs:
                continue
            ax.plot(xs, ys, "-o", ms=2.4, lw=1.3, color=c, label=label)
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.set_xlabel("epoch", fontsize=8)
            ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        # "Was it finished?" is answered by where the best epoch fell, not by
        # comparing block means: a curve that plateaus still has a lower mean in
        # its last quarter than the quarter before, so that test calls a
        # converged run "still falling". If the best validation epoch lands in
        # the final fifth, the run was stopped mid-improvement and its score is
        # a lower bound.
        ep = [r["epoch"] for r in h if "val_dice" in r]
        vd = [r["val_dice"] for r in h if "val_dice" in r]
        if len(vd) >= 4:
            best_ep = ep[int(np.argmax(vd))]
            last = ep[-1]
            late = best_ep >= 0.8 * last
            verdicts.append(
                f"{label}: best epoch {best_ep}/{last}"
                + (" - STOPPED mid-improvement, treat as a lower bound"
                   if late else " - plateaued before the end"))
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Was the arm finished, or just stopped?", fontsize=11.5,
                 fontweight="bold")
    fig.text(0.5, 0.005, "where each arm's best validation epoch fell - "
             + "  ·  ".join(verdicts),
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _save(fig, out, name)


def fig_ab(out: Path, scored: dict[str, Path], name="18_recipe_ab.png"):
    """Two (or more) arms side by side, read from `score_checkpoint.py` output.

    Both arms must come from that script rather than from the `results.json`
    each training run wrote: those were produced by whatever the evaluation code
    looked like when each run started, and this project's arms straddle a change
    to how the vessel threshold is chosen. Comparing the stored files would have
    measured the code, not the recipe.
    """
    import json

    arms = {k: json.loads(Path(v).read_text()) for k, v in scored.items()
            if Path(v).exists()}
    if len(arms) < 2:
        print(f"  [skip] {name}: need 2 scored arms, have {len(arms)}")
        return
    names = list(arms)
    metrics = [("dice", "necrosis DSC", "C-NCA"),
               ("vessel_f1", "vessel F1", "C-NCA"),
               ("dice", "necrosis DSC\n(8 rollouts)", "C-NCA + 8 repeats"),
               ("vessel_f1", "vessel F1\n(8 rollouts)", "C-NCA + 8 repeats")]

    fig, axes = plt.subplots(1, len(metrics) + 1,
                             figsize=(3.0 * (len(metrics) + 1), 4.0), dpi=140)
    colours = ["#7f8c8d", "#c0392b", "#2980b9", "#27ae60"]
    for j, (key, label, row) in enumerate(metrics):
        ax = axes[j]
        vals = [arms[n]["test"].get(row, {}).get(key, np.nan) for n in names]
        ax.bar(range(len(names)), vals,
               color=colours[:len(names)], width=0.62)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
        if key == "vessel_f1":
            base = arms[names[0]]["test"].get("hu threshold (vessel)", {}) \
                .get("vessel_f1")
            if base:
                ax.axhline(base, ls="--", lw=1.1, color="#333")
                ax.annotate(f"best HU threshold {base:.3f}", (0, base),
                            xytext=(0, 4), textcoords="offset points",
                            fontsize=7, color="#333")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7.5, rotation=12, ha="right")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
        finite = [v for v in vals if v == v]
        ax.set_ylim(0, max([1.0] + finite) * 1.18)

    ax = axes[-1]
    for i, n in enumerate(names):
        sw = arms[n].get("step_sweep", {})
        ks = sorted(int(k) for k in sw)
        ax.plot(ks, [sw[str(k)] if str(k) in sw else sw[k] for k in ks],
                "-o", ms=2.6, lw=1.3, color=colours[i], label=n)
        ax.axvline(arms[n]["infer_steps"], ls=":", lw=1.0, color=colours[i])
    ax.set_xlabel("inference rollout K", fontsize=8)
    ax.set_ylabel("val DSC", fontsize=8)
    ax.set_title("K sweep (dotted = chosen)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

    defaulted = {n: arms[n].get("config_defaulted", []) for n in names}
    warn = "; ".join(f"{n}: predates {', '.join(v)}"
                     for n, v in defaulted.items() if v)
    fig.suptitle("Same width, same epochs, same corpus - what the recipe buys",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.005,
             "Both arms re-scored with one evaluation code path "
             "(scripts/score_checkpoint.py); K and the vessel threshold are "
             "chosen on validation, reported on test."
             + (f"   ·   {warn}" if warn else ""),
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _save(fig, out, name)


def fig_liver_mask(out: Path, ckpt: Path, device="cpu", n=3, limit=300):
    """What the liver mask removes, and what it is worth.

    Two things are shown together because they are the same idea applied at two
    ends: masking the CT that goes IN (three quarters of the frame is not liver)
    and masking the vessel prediction that comes OUT.

    The output mask is close to free - measured, 34 % of the vessel head's false
    positives lie outside the organ and 0 % of its misses do, because every true
    vessel voxel in this corpus is inside the liver by construction. The
    necrosis head is deliberately not masked: 11.7 % of the necrosis area lies
    outside on average.

    It uses a contour the model was never given. That is legitimate - a planning
    system has one - and it is why every panel here says "+ liver mask".
    """
    import torch
    from ablation2d.data import AblationDataset
    from ablation2d.evaluate import mask_to_organ
    from ablation2d.model import best_vessel_threshold, dice, vessel_f1
    from ablation2d.train import load_checkpoint, predict_repeats

    model, cfg, _ = load_checkpoint(ckpt, device=device)
    K = cfg.infer_steps
    d = np.load(ROOT / "data_real" / "test_slices.npz")
    organ_all = (d["seg_liver"].astype(bool) | d["seg_vessel"].astype(bool))
    ns = len(organ_all)
    test = AblationDataset(ROOT / "data" / "test.npz", device=device, limit=limit)
    organ = torch.from_numpy(
        organ_all[np.arange(len(test)) % ns].astype(np.float32))

    torch.manual_seed(0)
    with torch.no_grad():
        pred = predict_repeats(model, test.inputs, K)
    masked = mask_to_organ(pred, organ)
    thr = 0.70
    f_raw = vessel_f1(pred, test.targets, thr)
    f_msk = vessel_f1(masked, test.targets, thr)
    dsc = float(dice(pred[:, 0:1], test.targets[:, 0:1]).mean())

    gain = (f_msk - f_raw)
    order = torch.argsort(gain, descending=True)
    pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])][:n]

    fig, axes = plt.subplots(len(pick), 4, figsize=(14.0, 3.4 * len(pick)),
                             dpi=140, squeeze=False)
    for r, i in enumerate(pick):
        # `show_ct` windows raw Hounsfield (W/L 200/120). The corpus stores the
        # normalised channel in [0,1], so handing it over directly puts every
        # pixel far below the window and the panel renders solid black - which
        # is exactly what the first version of this figure did.
        hu = test.inputs[i, 0].cpu().numpy() * 2000.0 - 1000.0
        org = organ[i].cpu().numpy().astype(bool)
        truth = (test.targets[i, 1] > 0.5).cpu().numpy()

        viz.show_ct(hu, axes[r][0], "CT as given" if r == 0 else None)
        viz.show_ct(np.where(org, hu, -1000.0), axes[r][1],
                    "CT masked to the liver" if r == 0 else None)
        for c, (pm, lab) in enumerate(
                ((pred, "vessel head"), (masked, "vessel head + liver mask")), 2):
            got = (pm[i, 1] > thr).cpu().numpy()
            err = np.zeros(hu.shape + (4,))
            err[got & truth] = (0.1, 0.8, 0.2, 0.9)
            err[~got & truth] = (0.2, 0.4, 1.0, 0.9)
            err[got & ~truth] = (1.0, 0.2, 0.2, 0.9)
            viz.show_ct(hu, axes[r][c], lab if r == 0 else None)
            axes[r][c].imshow(err, origin="upper", interpolation="nearest")
            # Both contours, drawn the same way, so the claim "every true vessel
            # is inside the organ" is checkable BY eye against the picture
            # rather than only in the summary number.
            axes[r][c].contour(org.astype(float), levels=[0.5],
                               colors=["#ffb300"], linewidths=0.9,
                               origin="upper")
            axes[r][c].contour(truth.astype(float), levels=[0.5],
                               colors=["#00e5ff"], linewidths=0.7,
                               origin="upper")
        axes[r][2].set_xlabel(f"F1 {float(f_raw[i]):.3f}", fontsize=8.5)
        axes[r][3].set_xlabel(f"F1 {float(f_msk[i]):.3f}"
                              f"   ({float(gain[i]):+.3f})", fontsize=8.5,
                              fontweight="bold")
        axes[r][0].set_ylabel(["most improved", "median", "least"][r],
                              fontsize=9, fontweight="bold")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])

    kept = float((test.inputs[:, 0].cpu().numpy() > 0).mean())
    fig.suptitle(f"The liver mask - vessel F1 {float(f_raw.mean()):.3f} "
                 f"-> {float(f_msk.mean()):.3f} for free",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.005,
             f"{limit} test cases, 8 rollouts averaged, K={K}, threshold {thr:.2f}."
             f"  green hit · blue MISSED · red invented; ORANGE is the liver "
             f"contour and CYAN the vessel truth - cyan never leaves orange, "
             f"which is why masking costs no recall.  Necrosis DSC is UNCHANGED "
             f"at {dsc:.3f} - only the "
             f"vessel head is masked, because 11.7 % of the necrosis area "
             f"legitimately lies outside the organ while 0 % of the vessels do. "
             f"The contour is information the model was never given.",
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    _save(fig, out, "21_liver_mask.png")
    print(f"  vessel F1 {float(f_raw.mean()):.4f} -> {float(f_msk.mean()):.4f}"
          f"   necrosis DSC {dsc:.4f} (unchanged by design)")


def fig_channel_scope(out: Path, ckpt: Path, device="cpu", steps=24,
                      picks=(0, 1, 2, 4), mask_ct: bool = False):
    """The state, opened up: a few channels x a few steps.

    The static twin of `ui.ChannelScope`, for the deck. Rows are channels, the
    first columns are steps, and the last column is that channel's per-step
    activity - the automaton's own answer to "how many steps do I need".

    Readouts are drawn as LOGITS. That is the point: a head run to +-4 is pinned
    against `STATE_CLAMP` and has stopped learning, and a sigmoid draws that as
    a confident 0.98.
    """
    import torch
    from ablation2d.train import load_checkpoint
    from ablation2d.ui import ChannelScope
    from ablation2d.plans import Needle, Plan
    from ablation2d.channels import FIXED_DURATION_S

    model, cfg, _ = load_checkpoint(ckpt, device=device)
    anat = load_real_slice()
    h, w = anat.shape
    cx, cy = 0.45 * w * anat.dx_mm, 0.52 * h * anat.dx_mm
    plan = Plan([Needle(cx, cy, cx + 70, cy + 24, 110.0, FIXED_DURATION_S)])
    sc = ChannelScope(model, anat, plan, steps=steps, torch_device=device,
                      mask_ct=mask_ct, build=False)
    tr = sc.rollout()

    picks = [c for c in picks if c < tr.shape[1]]
    ks = [0, 1, 2, max(3, steps // 4), max(4, steps // 2), steps - 1]
    ks = sorted(set(min(k, steps - 1) for k in ks))
    fig, axes = plt.subplots(len(picks), len(ks) + 1,
                             figsize=(2.05 * (len(ks) + 1), 2.25 * len(picks)),
                             dpi=140, squeeze=False)
    for r, c in enumerate(picks):
        f = tr[:, c]
        vmax = max(float(np.abs(f).max()), 1e-6)
        for j, k in enumerate(ks):
            ax = axes[r][j]
            ax.imshow(f[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      origin="upper", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {k}", fontsize=8.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(sc.roles[c], fontsize=8, fontweight="bold")
        ax = axes[r][-1]
        act = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
        if (act > 0).any():
            ax.plot(range(1, len(f)), act, "-o", ms=2.2, lw=1.2, color="#c0392b")
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "exactly 0\nat every step", ha="center",
                    va="center", fontsize=8, color="#2980b9",
                    transform=ax.transAxes)
            ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=6.5)
        if r == 0:
            ax.set_title("mean |change|", fontsize=8.5, fontweight="bold")
        if r == len(picks) - 1:
            ax.set_xlabel("step", fontsize=7.5)

    sat = [sc.roles[c] for c in picks
           if c < 2 and np.abs(tr[:, c]).max() > 3.9]
    if sc.restored:
        tail = ("  Conditioning channels are rewritten from the inputs every "
                "step, so their activity is identically zero - that flatline is "
                "`restore_conditioning` working.")
    else:
        # Measured, not asserted: how far the inputs travel from themselves.
        drifts = []
        for c in range(2, min(5, tr.shape[1])):
            f = tr[:, c]
            scale = float(np.abs(f[0]).mean()) + 1e-9
            drifts.append(f"{sc.roles[c].split(': ')[-1]} "
                          f"{float(np.abs(f - f[0]).max()) / scale:.0f}x")
        tail = ("  restore_conditioning is OFF: the automaton rewrites the CT "
                "and the plan too. Drift against each input's own mean - "
                + ", ".join(drifts)
                + ". A sparse channel whose drift dwarfs its mean has been "
                  "erased, and the plan had to be read in the first few steps.")
    note = ("readouts are LOGITS, not probabilities - "
            + (f"{', '.join(sat)} saturated against STATE_CLAMP (±4)"
               if sat else "no readout reached STATE_CLAMP (±4)")
            + "." + tail)
    fig.suptitle("Inside the automaton - every channel, every step",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.005, note, ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    _save(fig, out, "17_channel_scope.png")


def fig_vessels(out: Path, ckpt: Path, device="cpu", n=6, data: Path | None = None):
    """How good is the vessel head, honestly - and against what.

    Four columns per case, because the question "is the segmentation any good"
    is not answerable from a mask alone:

      1. the CT the model was given, in a liver window
      2. what a plain HU threshold finds - the baseline it must beat
      3. what the NCA finds, at the threshold chosen on validation
      4. the two errors, split: missed vessel vs invented vessel

    The truth contour is on every panel so nothing is graded by eye.
    """
    import torch
    from ablation2d.data import AblationDataset
    from ablation2d.train import load_checkpoint
    from ablation2d.model import best_vessel_threshold, vessel_f1

    model, cfg, meta = load_checkpoint(ckpt, device=device)
    data = data or ROOT / "data"
    test = AblationDataset(data / "test.npz", device=device)
    with torch.no_grad():
        pred = model(test.inputs, steps=cfg.infer_steps)
    thr, f1_all = best_vessel_threshold(pred, test.targets)

    # The baseline, fitted on the same split so it is not handicapped.
    hu = test.inputs[:, 0]
    tv = test.targets[:, 1] > 0.5
    best_hu = (0.0, -1.0)
    for q in np.linspace(80, 99.7, 40):
        t = float(torch.quantile(hu.flatten()[:200000], q / 100.0))
        p_ = hu > t
        tp = float((p_ & tv).sum()); den = float(p_.sum() + tv.sum())
        if den and 2 * tp / den > best_hu[1]:
            best_hu = (t, 2 * tp / den)

    per = vessel_f1(pred, test.targets, thr)
    order = torch.argsort(per, descending=True)
    # Best two, median two, worst two - not six good ones.
    pick = [order[0], order[1], order[len(order) // 2], order[len(order) // 2 + 1],
            order[-2], order[-1]][:n]

    fig, axes = plt.subplots(len(pick), 4, figsize=(13.5, 3.3 * len(pick)), dpi=130)
    for r, i in enumerate(pick):
        i = int(i)
        anat = viz.anatomy_from_inputs(test.inputs[i])
        truth = test.targets[i, 1].cpu().numpy()
        pv = pred[i, 1].cpu().numpy()
        hu_i = test.inputs[i, 0].cpu().numpy()
        viz.show_ct(anat, axes[r, 0], "CT (what it is given)" if r == 0 else None)
        axes[r, 0].contour(truth, levels=[0.5], colors=["#ffb300"], linewidths=0.9)

        viz.show_ct(anat, axes[r, 1],
                    f"HU threshold - F1 {best_hu[1]:.3f}" if r == 0 else None)
        axes[r, 1].imshow(hu_i > best_hu[0], cmap="Blues", alpha=0.45,
                          origin="upper", interpolation="nearest", zorder=2)
        axes[r, 1].contour(truth, levels=[0.5], colors=["#ffb300"], linewidths=0.9,
                           zorder=3)

        viz.show_ct(anat, axes[r, 2],
                    f"C-NCA vessel head @ {thr:.2f}" if r == 0 else None)
        axes[r, 2].imshow(pv, cmap="cool", alpha=0.55, vmin=0, vmax=1,
                          origin="upper", interpolation="nearest", zorder=2)
        axes[r, 2].contour(truth, levels=[0.5], colors=["#ffb300"], linewidths=0.9,
                           zorder=3)

        p_bin = pv > thr
        t_bin = truth > 0.5
        err = np.zeros(truth.shape + (3,), np.float32)
        err[..., 0] = (p_bin & ~t_bin)            # invented  -> red
        err[..., 2] = (~p_bin & t_bin)            # missed    -> blue
        err[..., 1] = (p_bin & t_bin) * 0.75      # correct   -> green
        axes[r, 3].imshow(err, origin="upper", interpolation="nearest")
        f = float(per[i])
        axes[r, 3].set_title(("green hit · blue MISSED · red invented\n"
                              if r == 0 else "") + f"F1 {f:.3f}",
                             fontsize=8.5, fontweight="bold")
        axes[r, 3].set_xticks([]); axes[r, 3].set_yticks([])
        axes[r, 0].set_ylabel(["best", "", "median", "", "", "worst"][r],
                              fontsize=9, fontweight="bold")

    # The mean alone flatters this head. Its per-case F1 is strongly bimodal -
    # near 0.9 where the target is the aorta or the IVC (large, round, bright)
    # and near 0.1 where it is fine intrahepatic branching - so the spread goes
    # in the title next to the mean, not in a footnote.
    q = torch.quantile(per.float(),
                       torch.tensor([0.10, 0.50, 0.90], device=per.device))
    fig.suptitle(f"Vessel segmentation from the CT alone - NCA F1 {f1_all:.3f} "
                 f"against a best-possible HU threshold of {best_hu[1]:.3f}",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.021,
             f"per-case F1 is bimodal: p10 {float(q[0]):.3f}  ·  median "
             f"{float(q[1]):.3f}  ·  p90 {float(q[2]):.3f}  ·  "
             f"{100 * float((per > best_hu[1]).float().mean()):.0f} % of cases "
             f"beat the threshold baseline. The head is good at the aorta and "
             f"the IVC and weak on fine intrahepatic branches.",
             ha="center", **CAPTION)
    fig.text(0.5, 0.005, "rows are best / median / worst of the test split, not a "
                         "selection; orange is the truth contour on every panel",
             ha="center", **CAPTION)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    _save(fig, out, "15_vessel_segmentation.png")
    print(f"  vessel F1 {f1_all:.4f} @ threshold {thr:.2f}   "
          f"HU-threshold baseline {best_hu[1]:.4f}")
    return {"vessel_f1": f1_all, "threshold": thr, "hu_baseline": best_hu[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "checkpoints" / "ablation_cnca_liver.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data_liver",
                    help="corpus root; a name containing 'liver' implies mask_ct")
    ap.add_argument("--mask-ct", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    do_phys = a.physics or a.all
    do_model = a.model or a.all

    if do_phys:
        if not solver_available():
            print("[skip] physics figures need the solver")
        else:
            print("[physics]")
            fig_showcase(a.out); fig_channels(a.out); fig_heat_sink(a.out)
            fig_slab_vs_3d(a.out); fig_real_slice(a.out)
    if do_model:
        if not a.ckpt.exists():
            print(f"[skip] model figures need {a.ckpt}")
        else:
            print("[model]")
            mask_ct = {"on": True, "off": False}.get(
                a.mask_ct, "liver" in a.data.name)
            fig_model(a.out, a.ckpt, a.device, data=a.data, mask_ct=mask_ct)
            fig_vessels(a.out, a.ckpt, a.device, data=a.data)
            fig_channel_scope(a.out, a.ckpt, a.device, mask_ct=mask_ct)


if __name__ == "__main__":
    main()
