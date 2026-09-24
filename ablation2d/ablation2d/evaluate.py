"""Scores, and the baselines that give them meaning.

"Predict zero" is no baseline: 96-99 % of every slice is background. The one a
clinician would defend is the device chart, a sphere sized from power and time
with the anatomy ignored. Beating it means the model learned something about
the anatomy. For the vessels, the baseline is the best single HU threshold.
"""
from __future__ import annotations

import numpy as np
import torch

from .channels import CHANNEL_NAMES, FIXED_DURATION_S
from .device import EMPRINT_HP
from .physics import DEAD_CONTOUR

ACT = CHANNEL_NAMES.index("applicator_activation")
DX_MM = 2.0
PX_CM2 = (DX_MM ** 2) / 100.0

#: r_mm = C * P_W ** alpha * t_s ** BETA. No default: fit it on the training
#: split with `fit_sphere_baseline` and pass the result.
SPHERE_PARAMS = None


# --------------------------------------------------------------------------- #
# the device-chart baseline
# --------------------------------------------------------------------------- #
def _slot_groups(x: np.ndarray):
    """Yield `(mask, dial_W, duration_s)` for each distinct applicator power.
    Needles at the same power share a disc radius, so they share a mask."""
    act = x[ACT]
    on = act > 0
    if not on.any():
        return
    for a in np.unique(act[on]):
        m = on & np.isclose(act, a)
        yield m, float(a) * EMPRINT_HP.max_power_w, FIXED_DURATION_S


def _distance_to(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask)
    except ImportError:
        ys, xs = np.nonzero(mask)
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        return np.min(np.hypot(yy[..., None] - ys, xx[..., None] - xs), axis=-1)


def sphere_baseline(inputs, params: dict | None = None) -> torch.Tensor:
    """(B, 3, H, W) inputs -> (B, 1, H, W): discs around each slot, anatomy ignored."""
    p = params or SPHERE_PARAMS
    if p is None:
        raise ValueError("the sphere baseline is not fitted: pass "
                         "params=fit_sphere_baseline(train_split)")
    arr = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)
    out = np.zeros((arr.shape[0], 1, *arr.shape[2:]), np.float32)
    for b in range(arr.shape[0]):
        for mask, dial_w, dur_s in _slot_groups(arr[b]):
            if dial_w <= 0:
                continue
            r_mm = p["C"] * (dial_w ** p["ALPHA"]) * (dur_s ** p["BETA"])
            out[b, 0] = np.maximum(out[b, 0],
                                   (_distance_to(mask) * DX_MM <= r_mm).astype(np.float32))
    t = torch.from_numpy(out)
    return t.to(inputs.device) if torch.is_tensor(inputs) else t


def fit_sphere_baseline(ds, max_samples: int = 600) -> dict:
    """Least squares for `log r = log C + a log P + b log t`.

    Fitted on single-setting cases only, so the radius is that of one lesion;
    it is then applied to every case, as a planner would use a device chart.
    """
    X, Y = [], []
    n = min(len(ds), max_samples)
    for i in range(n):
        xi, yi = ds.sample(i)
        x = xi[0].cpu().numpy()
        groups = list(_slot_groups(x))
        if len(groups) != 1:
            continue
        _, dial_w, dur_s = groups[0]
        if dial_w <= 0:
            continue
        area_px = float((yi[0, 0].cpu().numpy() >= DEAD_CONTOUR).sum())
        if area_px < 4:
            continue
        r_mm = np.sqrt(area_px * DX_MM * DX_MM / np.pi)
        X.append([1.0, np.log(dial_w), np.log(dur_s)])
        Y.append(np.log(r_mm))
    if len(X) < 20:
        raise RuntimeError(f"only {len(X)} usable single-setting cases to fit on")
    X, Y = np.asarray(X), np.asarray(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    return {"C": float(np.exp(coef[0])), "ALPHA": float(coef[1]),
            "BETA": float(coef[2]),
            "_fit": f"n={len(X)} single-setting cases, "
                    f"log-radius RMSE {float(np.sqrt(((pred - Y) ** 2).mean())):.4f}, "
                    f"R2 {float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()):.3f}"}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _death(t: torch.Tensor) -> torch.Tensor:
    return t[:, 0:1]


@torch.no_grad()
def score(pred: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5) -> dict:
    """Necrosis DSC, area error, recall and precision.

    Recall and precision are kept apart because under- and over-treatment are
    different mistakes, and DSC alone does not say which one the model makes.
    """
    p = (_death(pred) > threshold).flatten(1).float()
    t = (_death(truth) >= DEAD_CONTOUR).flatten(1).float()
    tp = (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    d = torch.where(den > 0, 2 * tp / den, torch.ones_like(den))
    area_err = (p.sum(1) - t.sum(1)) * PX_CM2
    has = t.sum(1) > 0
    return {
        "dice": float(d.mean()),
        "dice_p10": float(d.quantile(0.10)),
        "area_err_cm2": float(area_err.mean()),
        "area_abs_err_cm2": float(area_err.abs().mean()),
        "recall": float((tp[has] / t.sum(1)[has].clamp(min=1)).mean()),
        "precision": float((tp[has] / p.sum(1)[has].clamp(min=1)).mean()),
        "n": int(len(d)),
    }


@torch.no_grad()
def predict_all(model, ds, steps: int, batch_size: int = 16) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x, steps=steps) for x, _ in ds.batches(batch_size, shuffle=False)])


@torch.no_grad()
def mask_to_organ(pred: torch.Tensor, organ: torch.Tensor,
                  channels=(1,)) -> torch.Tensor:
    """Zero the given prediction channels (by default the vessels) outside the organ.

    This uses the liver contour, which the model is not given, so a score
    computed with it must be labelled as such. Every true vessel voxel lies in
    the organ, so masking the vessel head costs no recall. The necrosis head is
    not masked: part of a real lesion often extends beyond the liver.
    """
    out = pred.clone()
    m = organ.to(pred.device)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    for c in channels:
        out[:, c:c + 1] = out[:, c:c + 1] * m
    return out


def report(model, ds, steps: int, baselines: dict | None = None,
           batch_size: int = 16, vessel_threshold: float | None = None,
           tta: bool = False, average: str = "none") -> dict:
    """`{name: metrics}` for the model and every baseline on the same split.

    `vessel_threshold` should come from validation (see
    `model.best_vessel_threshold`). `average` is the test-time reduction:
    "none" (one stochastic rollout), "repeats" (mean of 8) or "d4" (mean over
    the 8 orientations). Report "repeats" next to "d4": most of the D4 gain is
    the averaging, not the orientations.
    """
    from .train import predict_d4_tta, predict_repeats

    if tta:
        average = "d4"
    truth = ds.targets if hasattr(ds, "targets") else ds.cell_death
    if average == "d4":
        pred = torch.cat([predict_d4_tta(model, x, steps)
                          for x, _ in ds.batches(batch_size, shuffle=False)])
    elif average == "repeats":
        pred = torch.cat([predict_repeats(model, x, steps)
                          for x, _ in ds.batches(batch_size, shuffle=False)])
    else:
        pred = predict_all(model, ds, steps, batch_size)
    out = {"C-NCA": score(pred, truth)}
    for name, fn in (baselines or {}).items():
        preds = torch.cat([fn(x) for x, _ in ds.batches(batch_size, shuffle=False)])
        out[name] = score(preds.to(truth.device), truth)
    out["predict zero"] = score(torch.zeros_like(truth), truth)

    if truth.shape[1] > 1:
        from .model import vessel_f1
        thr = 0.5 if vessel_threshold is None else float(vessel_threshold)
        f1 = vessel_f1(pred, truth, thr)
        out["C-NCA"]["vessel_f1"] = float(f1.mean())
        out["C-NCA"]["vessel_f1_p10"] = float(f1.quantile(0.10))
        out["C-NCA"]["vessel_threshold"] = thr
        out["hu threshold (vessel)"] = {"vessel_f1": hu_threshold_vessel_f1(ds)}
    return out


@torch.no_grad()
def hu_threshold_vessel_f1(ds, hu_channel: int = 0) -> float:
    """The best vessel F1 of a single Hounsfield threshold on this split.

    Fitted on the split it is scored on, which favours the baseline.
    """
    x = ds.inputs[:, hu_channel].flatten()
    t = (ds.targets[:, 1] > 0.5).flatten()
    best = 0.0
    for q in np.linspace(50, 99.5, 60):
        thr = float(torch.quantile(x[:200000], q / 100.0))
        p = x > thr
        tp = float((p & t).sum())
        den = float(p.sum() + t.sum())
        if den > 0:
            best = max(best, 2 * tp / den)
    return round(best, 4)
