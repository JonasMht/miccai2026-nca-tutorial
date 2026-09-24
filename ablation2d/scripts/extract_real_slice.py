#!/usr/bin/env python3
"""Pull one real axial liver slice out of my PhD project's corpus for the demo.

Run with a Python that can import the corpus reader module

    /path/to/training-codebase/.venv/bin/python \
        scripts/extract_real_slice.py --case case-1001 --out data/real_slice.npz

The result is a plain `.npz` with two boolean masks on the workshop's own
128 x 128 @ 2 mm grid, so the notebook needs neither the 3-D corpus package nor the 136 GB
corpus to open it. Nothing patient-identifying travels with it: two binary
masks and a case label, from a de-identified segmentation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CORPUS = Path("/home/jonas/Documents/Research/datasets/ablation_3d_1mm/train/patients")
NX = NY = 128
DX = 2.0


def block_reduce_max(a, f):
    """Downsample a mask by max over f x f blocks.

    Max, not mean: a 2 mm cell that contains any vessel IS a vessel cell. Mean
    pooling a 3 mm vein at 1 mm into 2 mm cells erases it, and the thin
    structures are the only ones the heat sink argument is about.
    """
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).max(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case-1001")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data" / "real_slice.npz")
    ap.add_argument("--slice", type=int, default=None,
                    help="axial index; default = the slice with the most vessel")
    ap.add_argument("--list-candidates", action="store_true")
    a = ap.parse_args()

    import unidata

    ds = unidata.open(str(a.corpus / a.case / "anatomy.unidata"), strict=False)
    liver = np.asarray(ds.read("seg_liver")) > 127        # (z, y, x), 1 mm
    vessel = np.asarray(ds.read("seg_vessel")) > 127
    spacing = ds.entity("seg_liver").spatial.spacing
    if not np.allclose(spacing, 1.0):
        raise SystemExit(f"expected 1 mm isotropic, got {spacing}")

    # Score every axial slice: we want liver and vessel, and enough of both that
    # the picture is worth putting on a screen.
    liv_a = liver.sum(axis=(1, 2))
    ves_a = vessel.sum(axis=(1, 2))
    score = np.where(liv_a > 0.35 * liv_a.max(), ves_a, 0)
    if a.list_candidates:
        for z in np.argsort(score)[::-1][:15]:
            print(f"  z={z:4d}  liver {liv_a[z]:6d} px  vessel {ves_a[z]:5d} px "
                  f"({100 * ves_a[z] / max(liv_a[z], 1):.1f} %)")
        return
    z = int(a.slice) if a.slice is not None else int(np.argmax(score))

    liv2, ves2 = liver[z], vessel[z]
    # Centre the 256 mm window on the liver's centroid, then downsample 1 -> 2 mm.
    ys, xs = np.nonzero(liv2)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = int(NX * DX / 2)                                # 128 mm at 1 mm
    pad = ((half, half), (half, half))
    liv_p = np.pad(liv2, pad); ves_p = np.pad(ves2, pad)
    y0, x0 = cy + half - half, cx + half - half
    win_l = liv_p[y0:y0 + 2 * half, x0:x0 + 2 * half]
    win_v = ves_p[y0:y0 + 2 * half, x0:x0 + 2 * half]

    # The two source labels are disjoint (vessel is carved out of parenchyma),
    # so they are pooled independently and the organ is their union. ANDing them
    # is the natural-looking mistake: it survives only on the boundary and
    # reported this slice as 3.2 % vessel against a true 13.4 %.
    f = int(DX)
    seg_liver = block_reduce_max(win_l, f)
    seg_vessel = block_reduce_max(win_v, f)
    organ = seg_liver | seg_vessel
    seg_liver = organ & ~seg_vessel

    meta = {
        "case": a.case, "axial_index": z, "source_spacing_mm": 1.0,
        "grid": [NY, NX], "dx_mm": DX,
        "organ_cm2": round(float(organ.sum()) * DX * DX / 100, 1),
        "vessel_pct_of_organ": round(100 * float(seg_vessel.sum())
                                     / max(float(organ.sum()), 1), 1),
        "note": ("de-identified segmentation, 3-D corpus from my PhD work; "
                 "masks only, no HU and no identifiers"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, seg_liver=seg_liver, seg_vessel=seg_vessel,
                        meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
