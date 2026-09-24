#!/usr/bin/env python3
"""Generate the 2-D ablation corpus.

    python scripts/gen_dataset.py --n 3000 --workers 12 --out data

OMP_NUM_THREADS is pinned to 1 before anything imports the solver. OpenMP reads it
when the runtime loads, so setting it later - inside the worker, say - is a
no-op, and 12 workers each spawning 32 threads is how you take a machine down
rather than how you make it fast.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d.generate import GenConfig, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--nz", type=int, default=None,
                    help="slab depth; default = ablation2d.physics.NZ")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--real-slices", type=Path, default=None,
                    help="directory of extracted patient slices; keeps the "
                         "corpus's PATIENT-level split")
    ap.add_argument("--plans-per-slice", type=int, default=2)
    a = ap.parse_args()
    from ablation2d.physics import NZ
    cfg = GenConfig(n_samples=a.n, workers=a.workers, seed=a.seed,
                    nz=a.nz if a.nz is not None else NZ, out=a.out,
                    real_slices=a.real_slices,
                    plans_per_slice=a.plans_per_slice)
    print(f"[gen] {cfg.n_samples} samples, {cfg.workers} workers, nz={cfg.nz} -> {cfg.out}")
    generate(cfg, progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
