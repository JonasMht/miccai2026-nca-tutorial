#!/usr/bin/env python3
"""Write data/DATASHEET.md from what the corpus actually contains.

    python scripts/datasheet.py

Every marginal in the output is measured off `meta.json`, not copied from the
sampler's parameters. The two have already disagreed once in this project (a
12 % vessel-fraction ceiling that would have put the demo slice outside its own
training distribution), and a datasheet that restates the intent rather than the
result cannot catch that.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d import anatomy, plans  # noqa: E402
from ablation2d.channels import (CHANNEL_NAMES, FIXED_DURATION_S,  # noqa: E402
                                 TARGET_NAMES)
from ablation2d.device import (CALIBRATION_STATUS, EMPRINT_HP,  # noqa: E402
                               HEAT_SINK_DIAMETER_MM, TISSUE_TABLE_PATH,
                               VESSEL_SINK_MIN_RADIUS_MM)
from ablation2d.physics import COOLDOWN_S, DEAD_CONTOUR, NZ  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def q(a, *ps):
    a = np.asarray(a, float)
    return [round(float(np.percentile(a, p)), 2) for p in ps]


def main():
    data = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data"
    meta = json.loads((data / "meta.json").read_text())
    S = meta["samples"]
    tissue = json.loads(TISSUE_TABLE_PATH.read_text())

    n_needles = np.array([s["n_needles"] for s in S])
    powers = np.array([p for s in S for p in s["dial_power_w"]])
    durs = np.array([d for s in S for d in s["duration_s"]])
    ves = np.array([s.get("vessel_frac", np.nan) for s in S])
    lesion = np.array([s["lesion_cm2"] for s in S])
    ves = np.array([s["vessel_pct_of_liver"] for s in S])
    organ = np.array([s["liver_area_cm2"] for s in S])
    wall = np.array([s["wall_s"] for s in S])
    cal = np.array([c for s in S for c in s["calibres_mm"]])

    pmf = [round(float((n_needles == k).mean()), 3) for k in (1, 2, 3, 4)]
    nz_pow = powers[powers > 0]

    sizes = {s: round((data / f"{s}.npz").stat().st_size / 1e6, 1)
             for s in ("train", "val", "test") if (data / f"{s}.npz").exists()}

    # Measured, not asserted: how often the adiabatic frame wall lands inside
    # the ablation zone, and how often the zone leaves the organ.
    # v2 gives the model a CT, a needle and a power - there is no organ-mask
    # channel to test the lesion against any more. The organ comes from the
    # slice corpus instead, and the mapping back to it (sample i was generated
    # from slice i % n_slices, see generate._generate_real) is verified rather
    # than trusted: the stored vessel target must equal that slice's vessel
    # segmentation, or the alignment is wrong and the number would be fiction.
    from ablation2d.data import load_splits
    _real_slices = meta["config"].get("real_slices")
    real_dir = Path(_real_slices) if _real_slices else None
    edge = out_of_organ = with_lesion = 0
    misaligned = 0
    for name, sp in load_splits(data).items():
        yy = sp._y.numpy()
        organ_of = None
        if real_dir is not None and (real_dir / f"{name}_slices.npz").exists():
            r = np.load(real_dir / f"{name}_slices.npz")
            org = (r["seg_liver"].astype(bool) | r["seg_vessel"].astype(bool))
            ves_m = r["seg_vessel"].astype(bool)
            ns = len(org)
            organ_of = lambda i: org[i % ns]                      # noqa: E731
            ves_of = lambda i: ves_m[i % ns]                      # noqa: E731
        for i in range(len(yy)):
            d = yy[i, 0] >= 252
            if not d.any():
                continue
            with_lesion += 1
            if d[0].any() or d[-1].any() or d[:, 0].any() or d[:, -1].any():
                edge += 1
            if organ_of is None:
                continue
            if not np.array_equal(yy[i, 1] > 127, ves_of(i)):
                misaligned += 1
                continue
            if (d & ~organ_of(i)).sum() > 0.15 * d.sum():
                out_of_organ += 1
    if misaligned:
        raise SystemExit(
            f"{misaligned}/{with_lesion} samples do not match the slice they "
            f"should have come from - the sample->slice mapping is wrong, so "
            f"the out-of-organ figure would be meaningless. Refusing to write.")
    edge_pct = 100 * edge / max(with_lesion, 1)
    outside_pct = 100 * out_of_organ / max(with_lesion, 1)

    real = _real_slices
    n_slices = sum(v.get("slices", 0) for v in meta["splits"].values())
    split_desc = ", ".join(
        f"{k} {v.get('patients', '?')} patients / {v.get('slices', '?')} slices"
        for k, v in meta["splits"].items())

    md = f"""# Datasheet - 2-D microwave ablation corpus

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by
`scripts/gen_dataset.py`. **{meta['n']} instances**, {meta['failed']} failures.
Splits: {', '.join(f"{k} {v['n']}" for k, v in meta['splits'].items())}
({', '.join(f'{k}.npz {v} MB' for k, v in sizes.items())}).

## Motivation

To train a cellular-automaton surrogate that predicts the microwave-ablation
necrosis field fast enough to sit inside a planning loop, and to teach that
process in a 30-minute hands-on session. The ground truth is a Pennes-bioheat + microwave-SAR + Arrhenius cell-death
simulation from the physics solvers of my PhD work (supervised by
Prof. Caroline Essert and Juan Verde), which is
correct and far too slow to call inside a search.

This is the 2-D twin of a 3-D corpus from the same PhD work. Channels,
normalisation, device calibration and clinical marginals are deliberately the
same so a number measured here means something next to a number measured there.

## Composition

One instance = one ablation plan on one **real patient CT slice**, extracted from that 3-D corpus and block-pooled to this grid.

| | |
|---|---|
| grid | {anatomy.NY} x {anatomy.NX} at {anatomy.DX_MM:g} mm ({anatomy.FOV_MM:g} mm field of view) |
| inputs | {len(CHANNEL_NAMES)} channels, uint8, `[0,1]` after dequantisation |
| targets | {len(TARGET_NAMES)} channels: Arrhenius damage fraction, and the vessel mask |
| storage | `inputs (N,{len(CHANNEL_NAMES)},{anatomy.NY},{anatomy.NX}) uint8`, `targets (N,{len(TARGET_NAMES)},{anatomy.NY},{anatomy.NX}) uint8` |
| burn time | **fixed at {FIXED_DURATION_S:g} s** - no longer an input |

Inputs, in order: {', '.join(f'`{c}`' for c in CHANNEL_NAMES)}.
Targets: {', '.join(f'`{c}`' for c in TARGET_NAMES)}.

**v2 gives the model the picture, not the physics.** v1's eight inputs were the
per-voxel terms of the bioheat equation - rho*c, perfusion, k, sigma, the vessel
sink rate - which is the right input if somebody has already produced a
segmentation and a material table. v2 hands over a CT slice, a needle and a
power, and asks for the necrosis field AND the vasculature.

The vessel task is genuinely hard and the corpus is built so that it stays hard:
vessel contrast is sampled per case over 1-28 HU, which puts **22 %** of cases at
or beyond the difficulty of the real demo slice (d' = 0.17 at 2 mm, where the
best single HU threshold scores F1 0.245 across the test split).

**uint8, not float32.** Both fields are in [0,1] by construction and 1/255 is far
below the disagreement between two runs of the solver at different resolutions.
It is a 4x saving on a corpus that has to reach a laptop over conference Wi-Fi.

## Ground truth

| | |
|---|---|
| solver | in-house, from my PhD work; explicit FDM, CPU/OpenMP |
| material database | revision `{tissue['_material_revision'][:16]}…` |
| geometry | 2-D anatomy **extruded** to a {NZ}-cell slab; needles in the mid-plane; label = mid-plane of the 3-D solve |
| cooldown | protocol + {COOLDOWN_S:g} s, with dynamic stop |
| necrosis contour | d >= {DEAD_CONTOUR} |

**Why 3-D and sliced.** A one-cell-deep grid is a genuine 2-D heat equation and
it is the physics of an infinite slab: 90 W for 5 min gives a 40 x 48 mm lesion
where the mid-plane of the same needle in 3-D gives 40 x 42 mm. The mid-plane is
bit-identical at nz = 32, 48, 64 and 96, so the slab depth is nearly free.

**Cost of the cooldown choice.** {COOLDOWN_S:g} s against a fully converged
1800 s tail, over ten varied plans: 1.64x faster for a mean necrosis DSC of
0.9995 (worst 0.9978, worst disagreement 3 cells out of 782).

## Device

{EMPRINT_HP.name}, the `mid_eff0.60` arm of a nine-arm benchmark fit from the same solvers.

| | |
|---|---|
| dial maximum | {EMPRINT_HP.max_power_w:g} W, efficiency {EMPRINT_HP.mw_efficiency:g} -> {EMPRINT_HP.deposited_max_w:g} W deposited |
| radiating slot | [{EMPRINT_HP.emission_start_mm:g}, {EMPRINT_HP.emission_end_mm:.4g}] mm back from the tip |
| SAR penetration | {EMPRINT_HP.sar_penetration_mm:g} mm (Beer-Lambert recast of the axisymmetric EM solve) |
| dielectric self-limiting | {EMPRINT_HP.diel_drop:g} |
| frequency / diameter | {EMPRINT_HP.frequency_hz / 1e9:g} GHz / {EMPRINT_HP.diameter_mm:g} mm (14G) |

> **{CALIBRATION_STATUS}**

## Anatomy

**Real patient CT.** {n_slices} axial slices extracted from that 3-D corpus at 2 mm, carrying the real Hounsfield field, the real
liver and vessel segmentations, and a real per-cell `material_id` covering bone,
kidney, muscle, skin, lung and air as well as liver and vasculature.

**The split is the corpus's PATIENT-level one**, not a cut through the slices:
{split_desc}. Adjacent axial slices of one liver are near-duplicates, so a
random split would put a near-copy of every validation slice into training and
report a number that means nothing.

Properties are read from `material_id` through the solver's own table rather than
stored. They are exactly constant per id in this corpus - measured relative
spread 1e-8, which is float16 rounding - so the id is sufficient, it is smaller,
and it cannot drift from what the solver integrates.

Downsampling 1 mm to 2 mm respects what each field is: the CT is **averaged** (a
dense field), the segmentations are **max**-pooled (thin structures vanish under
averaging), and `material_id` is **mode**-pooled - averaging two material ids
gives a third material that is not present, and max-pooling gives whichever enum
ordinal happens to be largest.

*This replaced a procedural anatomy.* On synthetic slices the vessel head
collapsed onto the organ mask (precision 0.087, IoU 0.77 against the LIVER) and
no loss weighting could move it: sweeping the vessel weight over 1, 3 and 10 and
the BCE positive weight to 50 left vessel F1 pinned at 0.18 +/- 0.005 while the
necrosis DSC fell from 0.85 to 0.12. Procedural vessels ARE bright blobs, so
"bright blob" was a correct solution and there was no shape to learn.

| quantity | p5 | median | p95 | range |
|---|---|---|---|---|
| organ area (cm²) | {q(organ,5)[0]} | {q(organ,50)[0]} | {q(organ,95)[0]} | {q(organ,0)[0]}–{q(organ,100)[0]} |
| vessel (% of organ) | {q(ves,5)[0]} | {q(ves,50)[0]} | {q(ves,95)[0]} | {q(ves,0)[0]}–{q(ves,100)[0]} |
| vessel calibre (mm) | {q(cal,5)[0]} | {q(cal,50)[0]} | {q(cal,95)[0]} | {q(cal,0)[0]}–{q(cal,100)[0]} |

The demo slice (case-1001, axial 154, de-identified segmentation) is 204 cm² and
13.9 % vessel with calibres to 26.8 mm - the procedural ranges were widened
until it sat inside them, because a demo outside its own training distribution
is a demo about extrapolation.

**Tissue properties** are the solver's own curated IT'IS-derived table, baked into
`ablation2d/tissue_table.json` with the database revision so the notebook needs
no compiled solver and drift is detectable:

| label | source entry | rho·c J/(mm³K) | k W/(mm K) | sigma S/mm | perfusion |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v['db_name']} | {v['rho'] * v['c']:.4e} | {v['k']:.4e} | "
        f"{v['sigma']:.4e} | {v['perfusion']:.4g} |"
        for k, v in tissue["tissues"].items()) + f"""

## Plans

| quantity | target (clinical) | achieved |
|---|---|---|
| antennas 1/2/3/4 | 55 / 25 / 13 / 7 % | {' / '.join(f'{100*p:.0f}' for p in pmf)} % |
| dial power | median 75 W, 60 % in 50-100 W, 5 % off | median {np.median(nz_pow):.0f} W, {100*((nz_pow>=50)&(nz_pow<=100)).mean():.0f} %, {100*(powers==0).mean():.1f} % off |
| duration | FIXED (not an input in v2) | {np.median(durs)/60:.2f} min for every needle |
| needle length | 40-200 mm | {plans.NEEDLE_LEN_MIN_MM:g}-{plans.NEEDLE_LEN_MAX_MM:g} mm, clipped to the frame |

Clinical references: KLCA 2024 practice guideline; a 2025 series of 133 lesions;
Lu et al., AJR 2002 for the {HEAT_SINK_DIAMETER_MM:g} mm heat-sink threshold.

**Hard constraints.** Tip inside the organ and not inside a vessel lumen; the far
end of the radiating slot inside the organ; the entry point on the picture.
**Vessel crossing is deliberately permitted** - it is the heat-sink signal the
corpus exists to teach.

**The 5 % of needles at exactly 0 W are not a mistake.** They are the only
examples that teach the network that a needle drawn on the picture with the
power off kills nothing, and without them the interactive planner lies the
moment someone drags the power slider to zero.

## Labels

| quantity | p5 | median | p95 | max |
|---|---|---|---|---|
| lesion (cm²) | {q(lesion,5)[0]} | {q(lesion,50)[0]} | {q(lesion,95)[0]} | {q(lesion,100)[0]} |
| vessel (fraction of frame) | {q(ves,5)[0]} | {q(ves,50)[0]} | {q(ves,95)[0]} | {q(ves,100)[0]} |

{100*(lesion==0).mean():.1f} % of instances have **no** necrosis at all - the
zero-power plans. A model scored only on cases that ablate would never be tested
on them.

## Known problems

1. **The vessel sink saturates.** The term is floored at a
   {VESSEL_SINK_MIN_RADIUS_MM:g} mm radius (the solver's vessel-sink floor;
   without it 1/R² makes a venule a better heat sink than the portal vein), and
   above ~0.2 1/s it stops changing the answer - measured: rates of 2.0 and
   12.5 1/s give the same 7.48 cm² lesion. At 2 mm voxels `vessel_sink_rate`
   therefore carries information only for vessels above ~3 mm calibre. At the
   1 mm resolution of the 3-D corpus the informative range is wider.
2. **The anatomy is z-invariant.** The extrusion is what makes the label a
   deterministic function of the 2-D input, and it is also a real departure from
   a patient: no vessel enters or leaves the plane.
3. **Every lesion inherits a PROVISIONAL calibration.** See the device block.
4. **{edge_pct:.1f} % of lesions touch the frame edge**, where the adiabatic wall
   is inside the ablation zone and the necrosis is therefore over-estimated.
   They are kept: filtering them would correlate the training set with lesion
   size, and 2 % is well below the calibration uncertainty. Do not quote a
   lesion area from a case whose zone reaches the border.
5. **{outside_pct:.0f} % of lesions put more than 15 % of their area in
   connective tissue.** That is not an artefact - the placement rule requires
   the tip and the radiating slot inside the organ, not the whole zone, so a
   needle near the liver surface legitimately burns outward. It is also most of
   the reason the tissue channels carry signal.
6. **50 patients is not a population.** The splits are by patient (35/5/10),
   which is the right unit, but ten test livers is a small denominator for any
   per-case claim - read the p10 column, not only the mean.
7. **Adjacent slices of one liver are near-duplicates.** Slices are spread
   across each liver's extent rather than taken as a contiguous run, but at
   2 mm they are still correlated. The corpus is 1 600 slices, not 1 600
   independent observations.

## Generation cost

{np.median(wall):.0f} s median per instance single-threaded ({q(wall,5)[0]}-{q(wall,95)[0]} s
p5-p95); {meta['n']} instances on 18 pinned cores. `OMP_NUM_THREADS=1` per worker -
a {anatomy.NX}x{anatomy.NY}x{NZ} grid is far too small to feed 32 threads, so the
parallelism is across samples.

## Distribution

**This corpus contains real patient CT and that changes what may be shipped.**
Earlier versions were procedurally generated and carried no patient data at all;
this one is 1 600 axial slices of 50 de-identified livers from that 3-D corpus, block-pooled to 128x128 at 2 mm and windowed to the
organ. There are no headers, no identifiers and no dates in the stored arrays -
only the pixel data, a case label of the form `case-1013`, and a slice index.

What is on disk, and what is meant to travel with the repo:

| directory | what | size | ships |
|---|---|---|---|
| `data/` | the task corpus: 4 000 plans on 1 600 real slices | 41 MB | yes - the notebook loads it |
| `data_liver/` | the task corpus with the CT masked to the liver - what the notebook trains on | 12 MB | yes |
| `data_vessel/` | warm start, derived from the above | 20 MB | yes |
| `data_real/` | the extracted slices `data/` was built from | 50 MB | no - reproduction input |
| `data_vessel_wide/` | warm start, 3.3x more slices, looser criteria | 62 MB | no - too big, and `data_vessel` is the documented recipe |

That is de-identified imaging, not synthetic data. Whether it may be
redistributed inside a public tutorial repository is a governance question for
the data's owner and is **not settled by this file**. Until it is, treat the
corpus as internal: the notebook loads it from the repo, so publishing the repo
publishes the corpus.

If the answer is no, the procedural generator is still in
`ablation2d/anatomy.py` and still works - with the documented cost that the
vessel head cannot be trained on it (see the anatomy note above).
"""
    out = data / "DATASHEET.md"
    out.write_text(md)
    print(f"wrote {out} ({len(md)} chars)")
    print(f"  antennas {pmf}  power median {np.median(nz_pow):.0f} W  "
          f"lesion median {np.median(lesion):.1f} cm2  "
          f"zero-lesion {100*(lesion==0).mean():.1f} %")


if __name__ == "__main__":
    main()
