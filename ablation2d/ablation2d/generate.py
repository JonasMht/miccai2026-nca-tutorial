"""Build the corpus: take a slice, sample a plan, solve, store.

The solver is OpenMP-parallel, but a 128 x 128 x 32 grid is too small to keep
many threads busy, so parallelism is across samples: many worker processes,
each with `OMP_NUM_THREADS=1`, one shard each, merged at the end.

Each split is one `.npz` with `inputs` (3 channels) and `targets` (necrosis,
vessels) stored as uint8. Both are in [0, 1], and 1/255 is well below the
solver's own resolution error. The real-CT corpus is 41 MB, small enough to
ship with the repo.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GenConfig:
    n_samples: int = 3000
    seed: int = 20260907
    out: Path = Path("data")
    workers: int = 12
    nz: int = 32
    splits: tuple = (("train", 0.85), ("val", 0.075), ("test", 0.075))
    #: Extracted real patient slices. When set, the split is by patient: adjacent
    #: slices of one liver are near-duplicates and must not straddle splits.
    real_slices: Path | None = None
    plans_per_slice: int = 2


#: Loaded once per worker process, not once per sample.
_SLICES: dict = {}


def _load_slices(path):
    import numpy as np
    key = str(path)
    if key not in _SLICES:
        z = np.load(path)
        _SLICES[key] = {k: z[k] for k in
                        ("hounsfield", "material_id", "seg_liver", "seg_vessel",
                         "case", "z")}
    return _SLICES[key]


def _one(seed: int, nz: int, real_slices=None, slice_index: int | None = None):
    """Generate a single (inputs, target, meta). Runs inside a worker."""
    import numpy as np
    from .anatomy import from_patient_slice, sample_anatomy
    from .channels import build_targets
    from .plans import sample_plan
    from . import physics

    rng = np.random.default_rng(seed)
    if real_slices is not None:
        d = _load_slices(real_slices)
        i = int(slice_index) % len(d["hounsfield"])
        base = dict(hounsfield=d["hounsfield"][i].astype(np.float32),
                    material_id=d["material_id"][i],
                    seg_liver=d["seg_liver"][i].astype(bool),
                    seg_vessel=d["seg_vessel"][i].astype(bool))
        case = str(d["case"][i]); zz = int(d["z"][i])
        for attempt in range(12):
            try:
                anat = from_patient_slice(**base, source=f"{case}@z{zz}")
                plan = sample_plan(rng, anat)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f"{case}@z{zz}: no feasible plan")
    else:
        for attempt in range(8):
            try:
                anat = sample_anatomy(rng)
                plan = sample_plan(rng, anat)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f"seed {seed}: no feasible plan in 8 attempts")

    s = physics.simulate(anat, plan, nz=nz)
    targets = build_targets(anat, s.cell_death)
    meta = {
        "seed": int(seed),
        "n_needles": plan.n,
        "dial_power_w": [round(n.dial_power_w, 1) for n in plan.needles],
        "duration_s": [round(n.duration_s, 1) for n in plan.needles],
        "lesion_cm2": round(s.lesion_cm2, 2),
        "vessel_frac": round(float(targets[1].mean()), 4),
        "wall_s": round(s.wall_s, 2),
        "sim_time_s": round(s.sim_time_s, 1),
        **anat.summary(),
        "plan": plan.as_dict(),
    }
    return (_q8(s.inputs), _q8(targets), meta)


def _q8(x):
    """[0,1] float -> uint8. See the module docstring for why this is enough."""
    return np.clip(np.rint(np.asarray(x, np.float32) * 255.0), 0, 255).astype(np.uint8)


def dequantise(u8):
    return np.asarray(u8, np.float32) / 255.0


def _worker(args):
    lo, hi, nz, seed0, real, n_slices = args
    os.environ["OMP_NUM_THREADS"] = "1"
    xs, ys, ms = [], [], []
    for i in range(lo, hi):
        try:
            x, y, m = _one(seed0 + i, nz, real,
                           (i % n_slices) if n_slices else None)
        except Exception as exc:                      # a bad draw must not kill the run
            ms.append({"seed": seed0 + i, "error": repr(exc)})
            continue
        xs.append(x); ys.append(y); ms.append(m)
    return np.stack(xs), np.stack(ys), ms


def _generate_real(cfg: GenConfig, progress=print):
    """One corpus per split, from the real slices, never mixing patients.

    The number of samples is `plans_per_slice` times the number of extracted
    slices - the anatomies are given, so the only thing sampled is the plan.
    """
    from concurrent.futures import ProcessPoolExecutor

    out_meta = {"splits": {}, "samples": [], "failures": []}
    for split in ("train", "val", "test"):
        src = Path(cfg.real_slices) / f"{split}_slices.npz"
        if not src.exists():
            continue
        n_slices = int(len(np.load(src)["z"]))
        n = n_slices * cfg.plans_per_slice
        chunk = max(1, n // (cfg.workers * 4))
        jobs = [(lo, min(lo + chunk, n), cfg.nz, cfg.seed + 100000 * len(split),
                 src, n_slices) for lo in range(0, n, chunk)]
        t0 = time.time()
        X, Y, M, done = [], [], [], 0
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            for x, y, m in ex.map(_worker, jobs):
                X.append(x); Y.append(y); M.extend(m); done += len(m)
                el = time.time() - t0
                progress(f"  {split} {done}/{n}  {el:5.0f}s  "
                         f"{el / max(done, 1):.2f} s/sample  "
                         f"ETA {(n - done) * el / max(done, 1) / 60:.1f} min")
        X = np.concatenate(X); Y = np.concatenate(Y)
        np.savez_compressed(cfg.out / f"{split}.npz", inputs=X, targets=Y)
        ok = [r for r in M if "error" not in r]
        cases = sorted({r.get("source", "?").split("@")[0] for r in ok})
        out_meta["splits"][split] = {
            "n": int(len(X)), "slices": n_slices, "patients": len(cases),
            "plans_per_slice": cfg.plans_per_slice}
        out_meta["samples"] += ok
        out_meta["failures"] += [r for r in M if "error" in r]
        progress(f"  wrote {split}.npz  n={len(X)}  from {len(cases)} patients  "
                 f"{(cfg.out / f'{split}.npz').stat().st_size / 1e6:.0f} MB")
    out_meta["n"] = sum(v["n"] for v in out_meta["splits"].values())
    out_meta["failed"] = len(out_meta["failures"])
    (cfg.out / "meta.json").write_text(json.dumps(
        {"config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in cfg.__dict__.items()}, **out_meta}, indent=1))
    progress(f"  total {out_meta['n']} samples, {out_meta['failed']} failures")
    return out_meta


def generate(cfg: GenConfig, progress=print):
    """Run the whole corpus and write `<out>/<split>.npz` + `datasheet.json`."""
    from concurrent.futures import ProcessPoolExecutor

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg.out.mkdir(parents=True, exist_ok=True)

    # real slices: one pass per split, keeping the corpus's PATIENT-level split.
    if cfg.real_slices is not None:
        return _generate_real(cfg, progress)

    chunk = max(1, cfg.n_samples // (cfg.workers * 4))
    jobs = [(lo, min(lo + chunk, cfg.n_samples), cfg.nz, cfg.seed, None, 0)
            for lo in range(0, cfg.n_samples, chunk)]

    t0 = time.time()
    X, Y, M = [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for x, y, m in ex.map(_worker, jobs):
            X.append(x); Y.append(y); M.extend(m)
            done += len(m)
            el = time.time() - t0
            progress(f"  {done}/{cfg.n_samples} samples  {el:6.0f}s elapsed  "
                     f"{el / max(done, 1):.2f} s/sample  "
                     f"ETA {(cfg.n_samples - done) * el / max(done, 1) / 60:.1f} min")
    X = np.concatenate(X); Y = np.concatenate(Y)
    ok = [m for m in M if "error" not in m]
    failed = [m for m in M if "error" in m]

    # Split by index, not by shuffling: every sample has its own independent
    # anatomy and plan, so there is no patient to leak across the boundary, and
    # a deterministic split means the val set is the same one next month.
    n = len(X)
    idx = np.arange(n)
    out_meta = {"n": n, "failed": len(failed), "splits": {}}
    lo = 0
    for name, frac in cfg.splits:
        hi = n if name == cfg.splits[-1][0] else lo + int(round(frac * n))
        sel = idx[lo:hi]
        np.savez_compressed(cfg.out / f"{name}.npz", inputs=X[sel], targets=Y[sel])
        out_meta["splits"][name] = {"n": int(len(sel)), "index": [int(lo), int(hi)]}
        progress(f"  wrote {name}.npz  n={len(sel)}  "
                 f"{(cfg.out / f'{name}.npz').stat().st_size / 1e6:.0f} MB")
        lo = hi

    (cfg.out / "meta.json").write_text(json.dumps(
        {"config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in cfg.__dict__.items()},
         **out_meta,
         "samples": ok, "failures": failed}, indent=1))
    progress(f"  total {time.time() - t0:.0f}s, {len(failed)} failures")
    return out_meta
