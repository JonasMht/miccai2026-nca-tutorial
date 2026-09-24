"""What the network is shown, and what it must produce.

    inputs   0  hounsfield              the CT slice, normalised to [0, 1]
             1  needle                  the applicator shaft
             2  applicator_activation   dial power / max power, at the radiating slot

    targets  0  cell_death              the Arrhenius damage field
             1  vessel                  the vessel mask

Every plan burns for 5 minutes, so duration is not an input.

The inputs are what a planning system has: a CT and a plan, no segmentation and
no tissue table. On the CT alone, vessels and parenchyma differ by little
(127 +/- 27 HU against 136 +/- 48 on the demo slice); what separates them is
shape, which a local rule iterated over many steps can pick up and a threshold
cannot.
"""
from __future__ import annotations

import numpy as np

from .anatomy import Anatomy, LBL_VESSEL, normalise_hu
from .device import EMPRINT_HP, Device
from .plans import Plan

#: The three things the model is given.
CHANNEL_NAMES = ["hounsfield", "needle", "applicator_activation"]
N_INPUT_CHANNELS = len(CHANNEL_NAMES)

#: The two answers, in the order the model holds them in its state.
TARGET_NAMES = ["cell_death", "vessel"]
N_TARGET_CHANNELS = len(TARGET_NAMES)

#: Every plan in the corpus burns for this long.
FIXED_DURATION_S = 300.0


# --------------------------------------------------------------------------- #
# painting the applicator
# --------------------------------------------------------------------------- #
def _segment_distance(shape, dx, a_mm, b_mm) -> np.ndarray:
    """Distance in mm from every cell centre to the segment a-b."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    px = (xx + 0.5) * dx
    py = (yy + 0.5) * dx
    ab = np.asarray(b_mm, np.float32) - np.asarray(a_mm, np.float32)
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return np.hypot(px - a_mm[0], py - a_mm[1]).astype(np.float32)
    t = ((px - a_mm[0]) * ab[0] + (py - a_mm[1]) * ab[1]) / L2
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (a_mm[0] + t * ab[0]),
                    py - (a_mm[1] + t * ab[1])).astype(np.float32)


def paint_plan(plan: Plan, shape, dx: float,
               device: Device = EMPRINT_HP) -> dict[str, np.ndarray]:
    """The needle and applicator channels of a plan.

    A 14G antenna (2.11 mm) is thinner than a 2 mm cell, so the paint radius is
    floored at half a cell; otherwise a needle between cell centres paints
    nothing. Overlapping needles take the max, not the sum: the channel states
    each applicator's own setting, and the solver deals with how they add up.
    """
    r_paint = max(0.5 * device.diameter_mm, 0.5 * dx)
    act = np.zeros(shape, np.float32)
    ndl = np.zeros(shape, np.float32)
    for n in plan.needles:
        d_shaft = _segment_distance(shape, dx, n.tip, n.base)
        ndl = np.maximum(ndl, (d_shaft <= r_paint).astype(np.float32))
        s0, s1 = n.slot_endpoints(device)
        d_slot = _segment_distance(shape, dx, s0, s1)
        slot = d_slot <= r_paint
        if not slot.any():                       # slot shorter than a cell
            slot = d_slot <= (d_slot.min() + 1e-6)
        act[slot] = np.maximum(act[slot], n.dial_power_w / device.max_power_w)
    return {"applicator_activation": act, "needle": ndl}


# --------------------------------------------------------------------------- #
# the input stack and the targets
# --------------------------------------------------------------------------- #
def build_inputs(anat: Anatomy, plan: Plan, *, device: Device = EMPRINT_HP,
                 mask_ct: bool = False) -> np.ndarray:
    """(3, NY, NX) float32: what the network sees.

    `mask_ct=True` zeroes the CT outside liver and vessels, the encoding of
    `data_liver/`. A model trained on that corpus must be fed the same way.
    """
    if anat.hounsfield is None:
        raise ValueError("this anatomy has no CT; pass hounsfield= to from_segmentations")
    painted = paint_plan(plan, anat.shape, anat.dx_mm, device)
    hu = normalise_hu(anat.hounsfield)
    if mask_ct:
        hu = hu * (anat.liver_mask | anat.vessel_mask)
    return np.stack([hu, painted["needle"], painted["applicator_activation"]]).astype(np.float32)


def build_targets(anat: Anatomy, cell_death: np.ndarray) -> np.ndarray:
    """(2, NY, NX) float32: necrosis, vessels."""
    vessel = (anat.label == LBL_VESSEL).astype(np.float32)
    return np.stack([np.asarray(cell_death, np.float32), vessel]).astype(np.float32)

