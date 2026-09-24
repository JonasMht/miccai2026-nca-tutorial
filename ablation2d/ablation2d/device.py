"""The applicator and the tissue table: the two things the label depends on
that are not in the picture.

Both come from the 3-D corpus of my PhD project, so the 2-D label is the same
physics, sliced:

* the applicator is the Medtronic Emprint HP as calibrated for the PhD solver
  (arm `mid_eff0.60` of a bench fit on ex vivo bovine liver at 17 C, held-out
  RMS 9.9 %, provisional, not clinically validated);
* the tissue properties are read from the solver's own IT'IS-derived table,
  so they cannot drift from it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .solver_env import require_solver

CALIBRATION_STATUS = (
    "Provisional: fitted 2026-07-28 on ex vivo bovine liver bench data at 17 C, "
    "held-out RMS 9.9 %, not clinically validated. Every lesion in this workshop "
    "inherits that status."
)


@dataclass(frozen=True)
class Device:
    """A calibrated microwave applicator."""

    name: str = "Medtronic Emprint HP (Thermosphere)"
    modality: str = "mw"

    # -- geometry, mm from the physical tip ---------------------------------
    diameter_mm: float = 2.11          # 14G
    emission_start_mm: float = 7.25    # near end of the radiating slot
    emission_length_mm: float = 11.8339  # the calibrated effective slot

    # -- drive ---------------------------------------------------------------
    frequency_hz: float = 2.45e9
    max_power_w: float = 150.0         # generator dial, not deposited power
    max_duration_s: float = 600.0
    mw_efficiency: float = 0.60        # pinned from independent evidence, not fitted

    # -- fitted SAR shape ----------------------------------------------------
    sar_penetration_mm: float = 9.2506

    # -- tissue coupling the corpus was generated with -----------------------
    diel_drop: float = 0.85            # my PhD project's fitted dielectric self-limiting

    @property
    def deposited_max_w(self) -> float:
        """What the solver actually puts into tissue at full dial."""
        return self.max_power_w * self.mw_efficiency

    def deposited_w(self, dial_w: float) -> float:
        return float(dial_w) * self.mw_efficiency

    @property
    def emission_end_mm(self) -> float:
        return self.emission_start_mm + self.emission_length_mm

    def apply(self, sim) -> None:
        """Install this device's fitted SAR shape on a Simulation.

        Without it the whole shaft radiates: on this grid the uncalibrated
        default gives a 28 x 56 mm cigar, the calibrated device 40 x 48 mm.
        """
        require_solver()          # fail clearly if the binding is absent
        sim.set_mw_calibration(
            penetration_mm=self.sar_penetration_mm,
            active_len_mm=self.emission_length_mm,
            diel_drop=self.diel_drop,
            vessel_sink=True,
        )


EMPRINT_HP = Device()


# --------------------------------------------------------------------------- #
# Tissue table
# --------------------------------------------------------------------------- #
# solver::Material enum ordinals (the solver types header). The id is what
# reaches the solver: it decides `is_biological_material` (which gates both the
# perfusion term and the vessel sink) and supplies q_met.
MAT_METAL = 1
MAT_VESSEL = 2
MAT_BLOOD = 3
MAT_LIVER = 11
MAT_TISSUE = 14      # generic soft tissue - our "connective tissue" background
MAT_FAT = 17

#: Which curated database entry each grid label is made of. Keeping the name
#: here and asking the solver for the numbers means the workshop cannot drift from
#: the engine's table the way a copied literal would.
TISSUE_OF = {
    "background": (MAT_TISSUE, "connective tissue"),
    "liver": (MAT_LIVER, "liver"),
    "vessel": (MAT_VESSEL, "blood"),
}


#: The baked table, next to this file. Written by scripts/bake_tissue_table.py
#: and carrying the material database's own revision hash, so a drift in
#: the solver's curated properties is detectable rather than silent.
TISSUE_TABLE_PATH = Path(__file__).with_name("tissue_table.json")


def query_solver_tissue(body_temp_k: float = 310.0) -> dict:
    """Ask the solver's own curated database. Needs the solver."""
    up = require_solver()
    out = {"_material_revision": up.material_revision(),
           "_solver_version": up.version_string(),
           "_body_temp_k": body_temp_k, "tissues": {}}
    for label, (mid, db_name) in TISSUE_OF.items():
        p = up.material_by_name(db_name, body_temp_k)
        out["tissues"][label] = {
            "db_name": db_name,
            "material_id": mid,
            "rho": float(p["rho"]),
            "c": float(p["c"]),
            "k": float(p["k"]),
            "sigma": float(p["sigma"]),
            "perfusion": float(p["perfusion"]),
            "curated": list(p["from_database"]),
        }
    return out


@lru_cache(maxsize=2)
def tissue_properties(body_temp_k: float = 310.0) -> dict[str, dict[str, float]]:
    """`{label: {rho, c, k, sigma, perfusion, material_id}}` in the solver units.

    rho kg/mm3, c J/(kg K), k W/(mm K), sigma S/mm, perfusion ml/(min kg).

    Reads the baked table, so the notebook needs no compiled solver.
    `verify_tissue_table` checks it against a live solver.
    """
    if not TISSUE_TABLE_PATH.exists():
        baked = query_solver_tissue(body_temp_k)
    else:
        baked = json.loads(TISSUE_TABLE_PATH.read_text())
        if baked.get("_body_temp_k") != body_temp_k:
            baked = query_solver_tissue(body_temp_k)
    return {k: v for k, v in baked["tissues"].items()}


def verify_tissue_table(body_temp_k: float = 310.0, rtol: float = 1e-6) -> str:
    """Compare the baked table with a live solver query; raise on any drift."""
    live = query_solver_tissue(body_temp_k)
    if not TISSUE_TABLE_PATH.exists():
        return "no baked table; nothing to verify"
    baked = json.loads(TISSUE_TABLE_PATH.read_text())
    bad = []
    for label, lv in live["tissues"].items():
        bv = baked["tissues"].get(label)
        if bv is None:
            bad.append(f"{label}: missing from the baked table")
            continue
        for f in ("rho", "c", "k", "sigma", "perfusion"):
            if abs(bv[f] - lv[f]) > rtol * max(abs(lv[f]), 1e-30):
                bad.append(f"{label}.{f}: baked {bv[f]!r} vs live {lv[f]!r}")
    if bad:
        raise RuntimeError(
            "the baked tissue table disagrees with the solver:\n  "
            + "\n  ".join(bad)
            + f"\nbaked revision {baked.get('_material_revision')}, "
              f"live {live['_material_revision']}. Re-bake and regenerate the "
              "corpus: channels and labels must come from one table.")
    return (f"baked table matches the solver "
            f"(material revision {live['_material_revision'][:12]}…)")


# --------------------------------------------------------------------------- #
# Vessel heat sink
# --------------------------------------------------------------------------- #
# Nu * k_b / (R^2 * rho c), the coefficient fdm_cpu.cpp actually integrates.
# Restated from solver::VESSEL_NUSSELT and solver::BLOOD_K.
VESSEL_NUSSELT = 3.66      # Graetz limit, laminar, constant wall temperature
BLOOD_K = 0.52e-3          # W/(mm K)

#: Lu et al., AJR 2002: heat sink in 32/44 hepatic veins > 3 mm diameter and
#: 0/20 below it. The corpus calls a vessel `heat_sink_capable` at calibre >= 3 mm.
HEAT_SINK_DIAMETER_MM = 3.0

#: Smallest vessel radius the convective sink is applied to (the solver's
#: floor); below it only Pennes perfusion and the blood material act. Without a
#: floor the 1/R^2 rate would make a venule a stronger sink than the portal
#: vein. The floor also matches where the term saturates on this grid: rates of
#: 2.0 and 12.5 1/s give the same lesion, and 0.2 1/s is already R = 1.58 mm.
#: VSINK_HI below is computed at this radius; change them together.
VESSEL_SINK_MIN_RADIUS_MM = 1.5

#: Encoding range for the vessel_sink_rate channel, from the PhD solver's
#: normalization_constants.h. Used to bring the channel into [0, 1].
VSINK_LO, VSINK_HI = 0.0, 0.60   # 1/s


def vessel_sink_rate(radius_mm: np.ndarray, rho_c: np.ndarray,
                     biological: np.ndarray) -> np.ndarray:
    """The sink rate in 1/s, gated exactly as the solver gates it: only on
    biological cells (air next to a vessel would otherwise blow up the rate)
    and only above `VESSEL_SINK_MIN_RADIUS_MM`. `physics.simulate` applies the
    same floor to the radius field it hands the solver.
    """
    r = np.asarray(radius_mm, np.float32)
    rc = np.asarray(rho_c, np.float32)
    ok = biological & (r >= VESSEL_SINK_MIN_RADIUS_MM) & (rc > 0.0)
    out = np.zeros_like(r, np.float32)
    np.divide(VESSEL_NUSSELT * BLOOD_K, np.where(ok, r * r * rc, 1.0),
              out=out, where=ok)
    return out


@lru_cache(maxsize=64)
def material_properties_by_id(material_id: int,
                              body_temp_k: float = 310.0) -> dict:
    """Properties for a solver::Material ordinal, from the engine's own table.

    Real slices carry per-cell materials beyond liver (bone, lung, muscle, fat,
    skin, air); reading them from the solver keeps both sides consistent.
    """
    up = require_solver()
    rho, c, k, sigma, perf = up.material_properties(int(material_id),
                                                    body_temp_k)
    return {"rho": float(rho), "c": float(c), "k": float(k),
            "sigma": float(sigma), "perfusion": float(perf),
            "material_id": int(material_id)}
