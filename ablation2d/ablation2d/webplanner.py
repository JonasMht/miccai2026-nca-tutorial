"""The planner that runs in the browser.

`BrowserPlanner` ships the model's weights and a few CT slices to the page, and
`webplanner.js` runs the NCA there as WebGL2 shaders. The automaton keeps
stepping on the viewer's GPU while the needles move, so dragging a needle shows
the field re-converge live, with no round trip to the Python kernel. That round
trip is what made the slider version (`ui.AblationStudio`) feel slow in Colab.

It works wherever the notebook's HTML output runs JavaScript: Colab, Jupyter,
VS Code, and the exported HTML.
"""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import numpy as np

from .anatomy import normalise_hu
from .device import EMPRINT_HP

JS = Path(__file__).with_name("webplanner.js")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _pack_conv(w: np.ndarray, out_tiles: int, in_tiles: int) -> np.ndarray:
    """(out, in, 3, 3) conv weight -> texels in the order the shader reads.

    Texel ((o*in_tiles + i)*9 + tap)*4 + row holds the 4 input channels of
    output channel 4*o + row, for input tile i at that tap.
    """
    co, ci = w.shape[:2]
    p = np.zeros((out_tiles * 4, in_tiles * 4, 3, 3), np.float32)
    p[:co, :ci] = w
    p = p.reshape(out_tiles, 4, in_tiles, 4, 3, 3)       # o, row, i, col, ky, kx
    return p.transpose(0, 2, 4, 5, 1, 3).reshape(-1)     # o, i, ky, kx, row, col


def export_model(model, name: str, vessel_threshold: float = 0.5, steps: int = 12) -> dict:
    """Weights and shape of a trained C-NCA, packed for the shader."""
    if getattr(model, "vessel_hidden", 0):
        raise ValueError("the browser planner reads the vessels off state channel 1; "
                         "a model with a separate vessel head is not supported")
    subs = list(model.sub_models)
    c = model.channels
    hidden = subs[0][0].out_channels
    t, ht = -(-c // 4), -(-hidden // 4)
    parts = []
    for sub in subs:
        w1 = sub[0].weight.detach().float().cpu().numpy()
        b1 = sub[0].bias.detach().float().cpu().numpy()
        w2 = sub[2].weight.detach().float().cpu().numpy()
        bias = np.zeros(ht * 4, np.float32)
        bias[:hidden] = b1
        parts += [_pack_conv(w1, ht, t), bias, _pack_conv(w2, t, ht)]
    roles = ["necrosis", "vessels", "CT", "needle", "power"] + [""] * (c - 5)
    return {
        "name": name, "channels": c, "hidden": hidden, "n_sub_models": len(subs),
        "fire_rate": float(getattr(model, "fire_rate", 1.0)), "vessel_threshold": float(vessel_threshold),
        "steps": int(steps), "anatomy_steps": int(getattr(model, "anatomy_steps", 0)),
        "params": int(sum(p.numel() for p in model.parameters())), "roles": roles,
        "weights": _b64(np.concatenate(parts).astype(np.float32)),
    }


def _deepest_point(mask: np.ndarray) -> tuple[int, int]:
    """The organ pixel farthest from the organ's edge (repeated erosion)."""
    m = mask.copy()
    last = np.argwhere(m)
    while m.any():
        last = np.argwhere(m)
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    y, x = last[len(last) // 2]
    return int(y), int(x)


def _best_hu_f1(ct: np.ndarray, organ: np.ndarray, vessel: np.ndarray) -> float:
    best = 0.0
    for q in np.linspace(50, 99.7, 60):
        p = (ct > np.percentile(ct[organ], q)) & organ
        den = p.sum() + vessel.sum()
        if den:
            best = max(best, 2 * float((p & vessel).sum()) / float(den))
    return best


def make_slice(name: str, ct01: np.ndarray, organ: np.ndarray, vessel: np.ndarray,
               dx_mm: float = 2.0) -> dict:
    """One patient slice. `ct01` is the CT normalised as the corpus stores it,
    (HU + 1000) / 2000; `organ` is liver | vessels."""
    organ = organ.astype(bool)
    vessel = vessel.astype(bool)
    y, x = _deepest_point(organ)
    return {
        "name": name,
        "ct": _b64(np.clip(np.round(ct01 * 255), 0, 255).astype(np.uint8)),
        "organ": _b64(np.packbits(organ)),
        "vessel": _b64(np.packbits(vessel)),
        "tip": [(x + 0.5) * dx_mm, (y + 0.5) * dx_mm],
        "angle": float(-np.pi / 2),              # from the anterior side, top of the image
        "hu_f1": _best_hu_f1(ct01, organ, vessel),
    }


def slice_from_anatomy(anat, name: str = "demo slice") -> dict:
    return make_slice(name, normalise_hu(anat.hounsfield), anat.liver_mask | anat.vessel_mask,
                      anat.vessel_mask, anat.dx_mm)


def slices_from_split(scanned, masked, n: int = 5, min_vessel_px: int = 150) -> list[dict]:
    """`n` test slices from different parts of the split, with visible vessels.

    `scanned` is the unmasked split (what the viewer sees), `masked` the same
    cases as the model was trained on; the organ mask is where the masked CT is
    nonzero. Consecutive cases share a slice, so candidates are spread out.
    """
    ct = scanned.inputs[:, 0].cpu().numpy()
    organ = masked.inputs[:, 0].cpu().numpy() > 0
    vessel = masked.targets[:, 1].cpu().numpy() > 0.5
    organ |= vessel
    ok = [i for i in range(0, len(ct), 2) if vessel[i].sum() >= min_vessel_px]
    pick = [ok[int(k)] for k in np.linspace(0, len(ok) - 1, n)]
    return [make_slice(f"test slice {i}", ct[i], organ[i], vessel[i]) for i in pick]


class BrowserPlanner:
    """Live planner for one or more trained models.

    Args:
        models: {"label": model} or a list of (label, model, vessel_threshold,
            steps), where steps is the rollout length K chosen on validation.
        slices: from `slice_from_anatomy` / `slices_from_split`.
        mask_ct: feed the model the CT zeroed outside the organ, as it was trained.
    """

    def __init__(self, models, slices, *, mask_ct: bool = True, device=EMPRINT_HP,
                 dx_mm: float = 2.0, grid: int = 128, max_needles: int = 2):
        if isinstance(models, dict):
            models = [(k, m, 0.5, 12) for k, m in models.items()]
        self.payload = {
            "grid": grid, "dx_mm": dx_mm, "mask_ct": mask_ct, "power_max_w": 250,
            "max_needles": max_needles,
            "device": {"diameter_mm": device.diameter_mm, "max_power_w": device.max_power_w,
                       "slot_start_mm": device.emission_start_mm,
                       "slot_end_mm": device.emission_end_mm},
            "models": [export_model(m, name, thr, k) for name, m, thr, k in models],
            "slices": list(slices),
        }

    def html(self) -> str:
        div = f"nca-planner-{uuid.uuid4().hex[:8]}"
        js = (JS.read_text()
              .replace("(ROOT, PAYLOAD)", f"(document.getElementById('{div}'), "
                                          f"{json.dumps(self.payload)})"))
        return f'<div id="{div}"></div>\n<script>\n{js}\n</script>'

    def _repr_html_(self):
        return self.html()

    def show(self):
        from IPython.display import HTML, display
        display(HTML(self.html()))
