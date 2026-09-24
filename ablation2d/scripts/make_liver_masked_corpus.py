#!/usr/bin/env python3
"""Rebuild the corpus with the CT masked to the liver - no re-simulation.

    python3 scripts/make_liver_masked_corpus.py --out data_liver

Only channel 0 (`hounsfield`) is masked. The needle and applicator channels are
left alone: the shaft enters through the body wall, and hiding the insertion
path would remove information the model legitimately has.

The targets are copied byte-for-byte. Masking is an input encoding, not a change
to the physics - the solver already ran on the full anatomy, and its answer must
not move. That is what makes this rebuild cost seconds instead of the ~19 s per
sample the simulation costs.

The sample -> slice mapping is `i % n_slices` (see generate._generate_real). It
is verified here rather than trusted, by checking each stored vessel target
against the slice it should have come from; a mismatch aborts, because a corpus
masked with the wrong slice's liver would train silently and score nonsense.

what this costs, measured on the test split before building it:

    mean necrosis area outside the liver mask   11.7 %
    cases with >15 % outside                    32.6 %
    cases with >50 % outside                     1.5 %

Outside the organ the corpus is not one material: 45.5 % background/soft tissue,
23.7 % air (frame padding), and ~25 % spread over five others. Masking keeps the
liver's shape (zeros mark the outside) but discards which tissue is out there,
so for the third of cases whose zone leaves the organ the model must predict the
overspill from an average rather than from the actual neighbour. Whether the
distractors it removes are worth more than the physics it hides is the
measurement, not something to assume.
"""
import argparse

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--slices", type=Path, default=ROOT / "data_real")
    ap.add_argument("--out", type=Path, default=ROOT / "data_liver")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src, sl = a.data / f"{split}.npz", a.slices / f"{split}_slices.npz"
        if not (src.exists() and sl.exists()):
            print(f"skip {split}"); continue
        z, d = np.load(src), np.load(sl)
        x, y = z["inputs"].copy(), z["targets"]
        organ = (d["seg_liver"].astype(bool) | d["seg_vessel"].astype(bool))
        ns = len(organ)

        bad = 0
        for i in range(len(x)):
            j = i % ns
            if not np.array_equal(y[i, 1] > 127, d["seg_vessel"][j].astype(bool)):
                bad += 1
                continue
            x[i, 0] = (x[i, 0] * organ[j]).astype(x.dtype)
        if bad:
            raise SystemExit(
                f"{split}: {bad}/{len(x)} samples do not match the slice they "
                f"should have come from - the sample->slice mapping is wrong "
                f"and the mask would be applied from another patient's liver.")

        p = a.out / f"{split}.npz"
        np.savez_compressed(p, inputs=x, targets=y)
        kept = float((x[:, 0] > 0).mean())
        print(f"{split}: {len(x)} samples, CT kept on {100*kept:.1f} % of cells "
              f"-> {p} ({p.stat().st_size/1e6:.1f} MB)", flush=True)

    for extra in ("meta.json", "real_slice.npz"):
        if (a.data / extra).exists():
            shutil.copy(a.data / extra, a.out / extra)
    (a.out / "MASKED").write_text(
        "inputs[:, 0] (hounsfield) multiplied by (seg_liver | seg_vessel).\n"
        "needle and applicator_activation are NOT masked.\n"
        "targets are byte-identical to ../data.\n")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
