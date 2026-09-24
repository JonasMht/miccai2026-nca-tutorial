#!/usr/bin/env python3
"""Execute both notebooks end to end and fail loudly on anything unexpected.

    python scripts/test_notebook.py --kernel nca-test          # all three modes

Needs `nbclient`, `nbformat`, `ipykernel` and `ipywidgets` in the kernel's
environment. Runs on CPU (CUDA hidden) with the notebook's QUICK switch, which
limits every split to 48 cases and training to 2 epochs - this checks that every
cell runs and that the pieces fit together, not that the model is good. The
model's quality is measured by the training arms, not here.

Three modes, because they exercise different code:

  train       SOLUTION, trains the model you wrote            -> zero errors
  pretrained  SOLUTION, loads the shipped checkpoint instead  -> zero errors
  tutorial    TUTORIAL, TODOs left blank                      -> the first error
              must be NotImplementedError, raised from a TODO cell
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"


def execute(path: Path, env: dict, kernel: str, timeout: int):
    nb = nbformat.read(path, as_version=4)
    saved = dict(os.environ)
    os.environ.update(env)
    try:
        client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                                allow_errors=True,
                                resources={"metadata": {"path": str(NB)}})
        t0 = time.time()
        client.execute()
        wall = time.time() - t0
    finally:
        os.environ.clear()
        os.environ.update(saved)
    rows = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
        rows.append({"cell": i, "src": c.source,
                     "error": (errs[0]["ename"], errs[0]["evalue"]) if errs else None,
                     "ran": c.get("execution_count") is not None})
    return nb, rows, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--timeout", type=int, default=900, help="per cell, seconds")
    ap.add_argument("--modes", default="train,pretrained,tutorial")
    ap.add_argument("--save", type=Path, default=None,
                    help="directory to write the executed notebooks into")
    ap.add_argument("--full", action="store_true",
                    help="drop the QUICK switch: full corpus, full epochs (slow on CPU)")
    a = ap.parse_args()

    base = {"CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    if not a.full:
        base["NCA_NOTEBOOK_QUICK"] = "1"
    plan = {
        "train": ("ablation_with_nca_SOLUTION.ipynb", {}),
        "pretrained": ("ablation_with_nca_SOLUTION.ipynb",
                       {"NCA_NOTEBOOK_PRETRAINED": "1"}),
        "tutorial": ("ablation_with_nca_TUTORIAL.ipynb", {}),
    }
    failed = False
    for mode in a.modes.split(","):
        name, extra = plan[mode]
        nb, rows, wall = execute(NB / name, {**base, **extra}, a.kernel, a.timeout)
        if a.save:
            a.save.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, a.save / f"{mode}.ipynb")
        errs = [r for r in rows if r["error"]]
        print(f"\n=== {mode}: {len(rows)} code cells, {len(errs)} with errors, {wall:.0f} s")
        if mode == "tutorial":
            if not errs:
                print("  FAIL: TUTORIAL ran clean - its TODOs are not blanks")
                failed = True
            else:
                first = errs[0]
                ok = (first["error"][0] == "NotImplementedError"
                      and "# TODO" in first["src"])
                print(f"  first error: cell {first['cell']} {first['error'][0]}"
                      f" -> {'OK (a TODO)' if ok else 'FAIL (not a TODO)'}")
                failed |= not ok
        else:
            for r in errs:
                line = next((l for l in r["src"].splitlines() if l.strip()), "")[:70]
                print(f"  cell {r['cell']:2d}  {r['error'][0]}: {r['error'][1][:160]}")
                print(f"           {line}")
            failed |= bool(errs)
    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
