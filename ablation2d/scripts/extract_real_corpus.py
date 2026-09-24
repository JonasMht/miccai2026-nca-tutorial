#!/usr/bin/env python3
"""Build the anatomy corpus from real patient CT, not procedural blobs.

Run with a Python that can import the corpus reader module

    /path/to/training-codebase/.venv/bin/python \
        scripts/extract_real_corpus.py --out data_real

Why this replaced the procedural anatomy: the vessel head trained on synthetic
slices learned "bright blob = vessel" and collapsed onto the organ mask
(precision 0.087, IoU 0.77 against the liver). That was a correct solution to
the synthetic task - procedural vessels are bright blobs. Real vasculature is a
branching tree with calibre that tapers, walls that partial-volume into
parenchyma, and a texture the parenchyma shares. There is a shape to learn.

**The split is by patient and comes from the corpus**, not by slice. Adjacent
axial slices of one liver are near-duplicates; splitting them randomly would put
a near-copy of every validation slice into training and report a number that
means nothing. 35 / 5 / 10 patients.

What is stored, per slice, on the workshop's 128 x 128 @ 2 mm grid:

    hounsfield    float16   the real CT
    material_id   uint8     solver::Material ordinal per cell
    seg_liver     uint8     parenchyma
    seg_vessel    uint8     vasculature

The five physical property fields are not stored: they are exactly constant per
`material_id` in this corpus (measured: relative spread 1e-8, i.e. float16
rounding), so keeping the id and reading the properties from the solver's own
table is smaller and cannot drift from what the solver integrates.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS = Path("/home/jonas/Documents/Research/datasets/ablation_3d_1mm")
NX = NY = 128
DX = 2.0
HU_LO, HU_HI = -1000.0, 1000.0

MAT_VESSEL, MAT_LIVER = 2, 11


def block_mean(a, f):
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def block_max(a, f):
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).max(axis=(1, 3))


def block_mode(a, f):
    """Majority vote. A material id is a label - averaging two ids gives a third
    material that is not there, and max-pooling gives whichever enum ordinal
    happens to be largest."""
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    blocks = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).transpose(0, 2, 1, 3)
    flat = blocks.reshape(h2 // f, w2 // f, f * f)
    out = np.zeros(flat.shape[:2], a.dtype)
    for i in range(flat.shape[0]):
        for j in range(flat.shape[1]):
            out[i, j] = Counter(flat[i, j].tolist()).most_common(1)[0][0]
    return out


def slices_for(ds, per_patient, min_liver_px, min_vessel_px):
    liver = np.asarray(ds.read("seg_liver")) > 127
    vessel = np.asarray(ds.read("seg_vessel")) > 127
    la = liver.sum(axis=(1, 2))
    va = vessel.sum(axis=(1, 2))
    ok = np.nonzero((la >= min_liver_px) & (va >= min_vessel_px))[0]
    if len(ok) == 0:
        return [], liver, vessel
    # Spread them across the liver's extent rather than taking a contiguous
    # run: consecutive 1 mm slices are near-identical and would make the corpus
    # look bigger than it is.
    idx = np.linspace(0, len(ok) - 1, min(per_patient, len(ok))).round().astype(int)
    return [int(ok[i]) for i in idx], liver, vessel


def extract(ds, z, liver, vessel, hu_vol, mid_vol):
    l2, v2 = liver[z], vessel[z]
    ys, xs = np.nonzero(l2 | v2)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = int(NX * DX / 2)
    pad = ((half, half), (half, half))

    def win(a, fill):
        return np.pad(a, pad, constant_values=fill)[cy:cy + 2 * half,
                                                    cx:cx + 2 * half]
    f = int(DX)
    hu = block_mean(win(hu_vol[z].astype(np.float32), -1000.0), f)
    mid = block_mode(win(mid_vol[z], 20), f)                 # 20 = air
    sv = block_max(win(v2.astype(np.uint8), 0), f) > 0
    sl = (block_max(win(l2.astype(np.uint8), 0), f) > 0) | sv

    # The anatomy must be self-consistent: the segmentation and the material id
    # are two views of the same thing, and they are pooled by different rules,
    # so force the vessel/liver ids to follow the pooled masks.
    mid = mid.copy()
    mid[sl & ~sv] = MAT_LIVER
    mid[sv] = MAT_VESSEL
    return hu.astype(np.float16), mid.astype(np.uint8), \
        sl.astype(np.uint8), sv.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "data_real")
    ap.add_argument("--per-patient", type=int, default=40)
    ap.add_argument("--min-liver-cm2", type=float, default=60.0)
    ap.add_argument("--min-vessel-cm2", type=float, default=3.0)
    a = ap.parse_args()

    import unidata

    a.out.mkdir(parents=True, exist_ok=True)
    min_liver = int(a.min_liver_cm2 * 100)          # at 1 mm, px == mm^2
    min_vessel = int(a.min_vessel_cm2 * 100)
    manifest = {}

    for split in ("train", "val", "test"):
        root = a.corpus / split / "patients"
        if not root.exists():
            continue
        HU, MID, SL, SV, CASE, Z = [], [], [], [], [], []
        for pdir in sorted(root.iterdir()):
            f = pdir / "anatomy.unidata"
            if not f.exists():
                continue
            ds = unidata.open(str(f), strict=False)
            zs, liver, vessel = slices_for(ds, a.per_patient, min_liver, min_vessel)
            if not zs:
                print(f"  {pdir.name}: no usable slice"); continue
            hu_vol = np.asarray(ds.read("hounsfield")) * 2000.0 - 1000.0
            mid_vol = np.asarray(ds.read("material_id"))
            for z in zs:
                hu, mid, sl, sv = extract(ds, z, liver, vessel, hu_vol, mid_vol)
                HU.append(hu); MID.append(mid); SL.append(sl); SV.append(sv)
                CASE.append(pdir.name); Z.append(z)
            print(f"  {pdir.name}: {len(zs)} slices", flush=True)
        if not HU:
            continue
        p = a.out / f"{split}_slices.npz"
        np.savez_compressed(p, hounsfield=np.stack(HU),
                            material_id=np.stack(MID),
                            seg_liver=np.stack(SL), seg_vessel=np.stack(SV),
                            case=np.array(CASE), z=np.array(Z))
        ves = np.stack(SV).mean()
        manifest[split] = {"n": len(HU), "patients": len(set(CASE)),
                           "vessel_frac": round(float(ves), 4),
                           "mb": round(p.stat().st_size / 1e6, 1)}
        print(f"{split}: {len(HU)} slices from {len(set(CASE))} patients "
              f"-> {p} ({manifest[split]['mb']} MB)", flush=True)

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
