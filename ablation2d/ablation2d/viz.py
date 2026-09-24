"""Figures, with one colour language for the whole session.

The CT is greyscale; necrosis is a warm ramp and vessels a blue one, both
transparent near zero so the anatomy shows through. Ground truth and prediction
always use the same ramp.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from .anatomy import DX_MM, LBL_BACKGROUND, LBL_LIVER, LBL_VESSEL, Anatomy
from .channels import CHANNEL_NAMES
from .device import EMPRINT_HP, HEAT_SINK_DIAMETER_MM

#: Necrosis: transparent -> amber -> deep red. Transparent below 0.1, because a
#: trained model leaves a small nonzero floor on background (median ~0.05) that
#: would otherwise render as speckle.
NECROSIS = mcolors.LinearSegmentedColormap.from_list("necrosis", [
    (0.00, (0.99, 0.85, 0.30, 0.00)),
    (0.10, (0.99, 0.85, 0.30, 0.00)),
    (0.22, (0.99, 0.75, 0.20, 0.55)),
    (0.55, (0.94, 0.42, 0.08, 0.85)),
    (1.00, (0.65, 0.06, 0.10, 0.95)),
])

#: Vessels: transparent -> light blue, same transparent floor.
VESSELS = mcolors.LinearSegmentedColormap.from_list("vessels", [
    (0.00, (0.25, 0.44, 0.67, 0.00)),
    (0.30, (0.25, 0.44, 0.67, 0.00)),
    (0.60, (0.36, 0.60, 0.90, 0.75)),
    (1.00, (0.88, 0.96, 1.00, 0.95)),
])

#: Label maps: background, liver, vessel.
TISSUE_COLOURS = {LBL_BACKGROUND: "#2a2a30", LBL_LIVER: "#7d6a5a", LBL_VESSEL: "#3f6fa8"}
ANATOMY_CMAP = mcolors.ListedColormap([TISSUE_COLOURS[k] for k in (0, 1, 2)])

NEEDLE_COLOUR = "#e8e8ee"
SLOT_COLOUR = "#39d98a"


def anatomy_from_inputs(inputs, dx_mm=None) -> Anatomy:
    """A rough organ outline recovered from a stored CT channel, for figures.

    Liver vs surroundings is a threshold on HU; vessels are left out (finding
    them is the model's job) and radii are unknown, so never feed this back
    into the solver.
    """
    x = np.asarray(inputs.detach().cpu() if hasattr(inputs, "detach") else inputs)
    if x.ndim == 4:
        x = x[0]
    hu = x[CHANNEL_NAMES.index("hounsfield")] * 2000.0 - 1000.0
    label = np.full(hu.shape, LBL_BACKGROUND, np.uint8)
    label[hu > 85.0] = LBL_LIVER
    return Anatomy(label=label, vessel_radius_mm=np.zeros(hu.shape, np.float32),
                   hounsfield=hu.astype(np.float32), dx_mm=dx_mm or DX_MM,
                   source="recovered from channels")


def show_anatomy(anat: Anatomy, ax=None, title=None):
    ax = ax or plt.gca()
    ax.imshow(anat.label, cmap=ANATOMY_CMAP, vmin=0, vmax=2,
              origin="upper", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return ax


def draw_plan(plan, anat: Anatomy, ax=None, device=EMPRINT_HP, label_power=True):
    """Shaft in white, radiating slot in green (grey when off), tip as a dot."""
    ax = ax or plt.gca()
    dx = anat.dx_mm
    for n in plan.needles:
        tp, bp = n.tip / dx - 0.5, n.base / dx - 0.5
        ax.plot([tp[0], bp[0]], [tp[1], bp[1]], color=NEEDLE_COLOUR,
                lw=1.6, solid_capstyle="round", zorder=4)
        s0, s1 = n.slot_endpoints(device)
        s0, s1 = s0 / dx - 0.5, s1 / dx - 0.5
        on = n.dial_power_w > 0
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=SLOT_COLOUR if on else "#888894",
                lw=4.0, solid_capstyle="butt", zorder=5, alpha=0.95)
        ax.plot(tp[0], tp[1], "o", ms=3.2, color=NEEDLE_COLOUR, zorder=6)
        if label_power:
            ax.annotate(f"{n.dial_power_w:.0f} W · {n.duration_s / 60:.1f} min" if on else "off",
                        xy=(tp[0], tp[1]), xytext=(4, -9), textcoords="offset points",
                        fontsize=6.5, color="w", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="#00000099", ec="none"))
    return ax


def show_ct(anat_or_hu, ax=None, title=None, window=(120.0, 200.0)):
    """The CT in a liver window (level 120, width 200 HU), where vessels are visible."""
    ax = ax or plt.gca()
    hu = getattr(anat_or_hu, "hounsfield", None)
    hu = np.asarray(anat_or_hu if hu is None else hu, np.float32)
    lvl, wid = window
    ax.imshow(hu, cmap="gray", vmin=lvl - wid / 2, vmax=lvl + wid / 2,
              origin="upper", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return ax


def show_necrosis(field, ax=None, anat=None, plan=None, title=None,
                  contour=0.99, contour_colour="#ffffff", on_ct=False):
    """Anatomy (or CT) underneath, the damage field on top, the dead-tissue contour."""
    ax = ax or plt.gca()
    if anat is not None:
        if on_ct and getattr(anat, "hounsfield", None) is not None:
            show_ct(anat, ax)
        else:
            show_anatomy(anat, ax)
    f = np.asarray(field)
    im = ax.imshow(f, cmap=NECROSIS, vmin=0, vmax=1, origin="upper",
                   interpolation="nearest", zorder=2)
    if contour is not None and (f >= contour).any():
        ax.contour(f, levels=[contour], colors=[contour_colour], linewidths=0.9, zorder=3)
    if plan is not None and anat is not None:
        draw_plan(plan, anat, ax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def show_channels(inputs, figsize=(10, 3.9)):
    """The three input channels, each scaled to its own maximum (printed below it),
    since the applicator channel covers only a few cells."""
    inputs = np.asarray(inputs)
    fig, axes = plt.subplots(1, len(CHANNEL_NAMES), figsize=figsize, dpi=110)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ch = inputs[i]
        hi = float(ch.max())
        ax.imshow(ch, cmap="viridis", vmin=0, vmax=max(hi, 1e-6),
                  origin="upper", interpolation="nearest")
        ax.set_title(f"{i}  {CHANNEL_NAMES[i]}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel(f"max {hi:.3f} · {int((ch > 0).sum())} cells > 0",
                      fontsize=6.5, color="#555", labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("What the network sees: a CT slice, a needle and a power",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes


def show_rollout(trace, every=1, max_frames=8, anat=None, figsize=(14, 2.4)):
    """The necrosis field, step by step."""
    idx = list(range(0, len(trace), every))[:max_frames]
    fig, axes = plt.subplots(1, len(idx), figsize=figsize, dpi=110)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, idx):
        f = trace[k]
        f = f.detach().cpu().numpy()[0, 0] if hasattr(f, "detach") else np.asarray(f)
        if anat is not None:
            show_anatomy(anat, ax)
        ax.imshow(f, cmap=NECROSIS, vmin=0, vmax=1, origin="upper",
                  interpolation="nearest", zorder=2)
        ax.set_title(f"step {k + 1}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig, axes


def vessel_legend_text(anat: Anatomy) -> str:
    r = anat.vessel_radius_mm[anat.vessel_mask]
    if not r.size:
        return "no vessels"
    cal = 2 * r
    n_sink = int((cal >= HEAT_SINK_DIAMETER_MM).sum())
    return (f"vessels {100 * anat.vessel_fraction:.0f} % of organ · "
            f"calibre {cal.min():.1f}-{cal.max():.1f} mm · "
            f"{100 * n_sink / len(cal):.0f} % of vessel area can sink heat "
            f"(>= {HEAT_SINK_DIAMETER_MM:.0f} mm)")


def show_case(x, y, hu_scanned=None, figsize=(13, 3.6)):
    """One corpus case: the CT, what the model is given, and the two targets.

    `x` is (3, H, W) inputs, `y` (2, H, W) targets; `hu_scanned` the same slice
    before masking, if you have it.
    """
    x, y = np.asarray(x), np.asarray(y)
    hu = x[0] * 2000.0 - 1000.0
    n = 3 + (hu_scanned is not None)
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=100)
    ax = list(ax)
    if hu_scanned is not None:
        show_ct(hu_scanned, ax.pop(0), "the CT as scanned")
    show_ct(hu, ax[0], f"input: masked CT, needle, {150 * x[2].max():.0f} W")
    ax[0].imshow(np.ma.masked_equal(x[1], 0), cmap=mcolors.ListedColormap([NEEDLE_COLOUR]),
                 origin="upper", interpolation="nearest")
    ax[0].imshow(np.ma.masked_equal(x[2], 0), cmap=mcolors.ListedColormap([SLOT_COLOUR]),
                 origin="upper", interpolation="nearest")
    show_ct(hu, ax[1], "target: necrosis")
    ax[1].imshow(y[0], cmap=NECROSIS, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    if (y[0] >= 0.99).any():
        ax[1].contour(y[0], levels=[0.99], colors=["white"], linewidths=0.9)
    show_ct(hu, ax[2], "target: vessels")
    ax[2].imshow(np.ma.masked_less(y[1], 0.5), cmap=VESSELS, vmin=0, vmax=1,
                 origin="upper", interpolation="nearest")
    fig.tight_layout()
    return fig


def vessel_gallery(ct, truth, pred, threshold, rows=("best", "median", "worst"), f1=None):
    """Vessel head on a few cases: CT, prediction over truth, and the error map.

    Green is vessel found, blue missed, red invented.
    """
    fig, axes = plt.subplots(len(ct), 3, figsize=(10.5, 3.5 * len(ct)), dpi=100,
                             squeeze=False)
    for r, (c, t, p) in enumerate(zip(ct, truth, pred)):
        hu = np.asarray(c) * 2000.0 - 1000.0
        t = np.asarray(t) > 0.5
        got = np.asarray(p) > threshold
        show_ct(hu, axes[r, 0], "CT (masked)" if r == 0 else None)
        axes[r, 0].set_ylabel(rows[r], fontsize=11, fontweight="bold")
        show_ct(hu, axes[r, 1], "vessel head, truth in orange" if r == 0 else None)
        axes[r, 1].imshow(np.ma.masked_less(np.asarray(p), threshold), cmap=VESSELS,
                          vmin=0, vmax=1, origin="upper", interpolation="nearest")
        axes[r, 1].contour(t, levels=[0.5], colors=["#ffb300"], linewidths=0.9)
        err = np.zeros(hu.shape + (4,))
        err[got & t] = (0.1, 0.8, 0.2, 0.9)
        err[~got & t] = (0.2, 0.4, 1.0, 0.9)
        err[got & ~t] = (1.0, 0.2, 0.2, 0.9)
        show_ct(hu, axes[r, 2], "found / missed / invented" if r == 0 else None)
        axes[r, 2].imshow(err, origin="upper", interpolation="nearest")
        if f1 is not None:
            axes[r, 2].set_xlabel(f"F1 {f1[r]:.3f}", fontsize=9)
    fig.tight_layout()
    return fig
