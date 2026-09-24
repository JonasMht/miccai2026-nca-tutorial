"""Ground truth: Pennes bioheat, microwave SAR and Arrhenius cell death, from the solver.

The label is solved in 3-D and then sliced. A one-cell-deep 2-D grid is an
infinite slab, with no heat leaving through the faces: at 90 W for 5 min it
gives a 40 x 48 mm lesion where the 3-D mid-plane gives 40 x 42 mm. So the
slice is extruded along z, the needles lie in the mid-plane, and the label is
the mid-plane of the 3-D solve, a deterministic function of the 2-D picture.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .anatomy import Anatomy
from .channels import build_inputs
from .device import EMPRINT_HP, VESSEL_SINK_MIN_RADIUS_MM, Device
from .plans import Plan
from .solver_env import require_solver

#: Slab depth, 32 cells = 64 mm. The mid-plane is identical from 32 cells up.
NZ = 32

#: Simulated time after the applicators switch off; the lesion keeps growing
#: meanwhile. 900 s agrees with 1800 s to a mean DSC of 0.9995 and is 1.6x faster.
COOLDOWN_S = 900.0

#: Arrhenius damage above which tissue counts as dead.
DEAD_CONTOUR = 0.99


@dataclass
class Sample:
    inputs: np.ndarray        # (3, NY, NX) float32, normalised
    cell_death: np.ndarray    # (NY, NX)   float32 in [0, 1]
    anatomy: Anatomy
    plan: Plan
    wall_s: float
    sim_time_s: float

    @property
    def lesion_cm2(self) -> float:
        px = self.anatomy.dx_mm ** 2
        return float((self.cell_death >= DEAD_CONTOUR).sum()) * px / 100.0


def _extrude(field2d: np.ndarray, nz: int) -> np.ndarray:
    """(NY, NX) -> flat (nz*NY*NX,) in the solver's x-fastest order.

    the solver indexes `i = x + nx*(y + ny*z)`, so a C-order (nz, ny, nx) array
    ravels to exactly that. Getting this backwards silently transposes the
    anatomy, which looks plausible and is wrong.
    """
    return np.repeat(field2d[None, :, :], nz, axis=0).ravel().copy()


def simulate(anat: Anatomy, plan: Plan, *, device: Device = EMPRINT_HP,
             nz: int = NZ, cooldown_s: float = COOLDOWN_S,
             return_volume: bool = False) -> Sample:
    """Solve one plan and return the mid-plane necrosis field."""
    ny, nx = anat.shape
    dx = anat.dx_mm
    cz = nz // 2

    up = require_solver()
    phys = anat.physical_channels()
    sim = up.Simulation(up.GridDef(nx, ny, nz, dx), "fdm")
    up._from_channels(
        sim,
        _extrude(phys["material_id"], nz).astype(np.uint8),
        _extrude(phys["rho"], nz), _extrude(phys["c"], nz),
        _extrude(phys["k"], nz), _extrude(phys["sigma"], nz),
        _extrude(phys["perfusion"], nz),
        nx, ny, nz, dx,
        _extrude((anat.vessel_radius_mm > 0).astype(np.uint8), nz).astype(np.uint8),
    )
    # The solver's vessel sink reads the radius field, not the vessel flag, so
    # it must be set explicitly, with the same floor as the channel
    # (device.VESSEL_SINK_MIN_RADIUS_MM).
    sink_radius = np.where(anat.vessel_radius_mm >= VESSEL_SINK_MIN_RADIUS_MM,
                           anat.vessel_radius_mm, 0.0).astype(np.float32)
    up._set_vessel_radius(sim, _extrude(sink_radius, nz))

    device.apply(sim)

    events = []
    for n in plan.needles:
        e = up.SimulationEvent()
        e.applicator_type = up.ApplicatorType.MW
        e.energy = n.deposited_w(device)     # watts into tissue, not the dial
        e.duration = n.duration_s
        e.start_time = 0.0
        # p1 is the tip, p2 the base: the radiating slot is measured back from p1
        e.p1 = up.Vec3(float(n.tip_x_mm), float(n.tip_y_mm), (cz + 0.5) * dx)
        e.p2 = up.Vec3(float(n.base_x_mm), float(n.base_y_mm), (cz + 0.5) * dx)
        e.frequency = device.frequency_hz
        e.diameter = device.diameter_mm
        events.append(e)
    sim.set_schedule_mode(True)              # every antenna fires from t=0
    sim.set_events(events)
    sim.set_dynamic_stop(True, 0.0, 0)

    horizon = max((n.duration_s for n in plan.needles), default=0.0) + cooldown_s
    t0 = time.time()
    sim.run(horizon)
    wall = time.time() - t0

    frame = sim.capture()
    dead = np.asarray(frame.dead, np.float32).reshape(nz, ny, nx)
    mid = dead[cz].copy()

    s = Sample(inputs=build_inputs(anat, plan, device=device),
               cell_death=mid, anatomy=anat, plan=plan,
               wall_s=wall, sim_time_s=float(frame.sim_time))
    if return_volume:
        s.volume = dead                       # type: ignore[attr-defined]
    return s
