#!/usr/bin/env python3
"""Freeze the solver's curated tissue properties into a JSON the notebook can read.

    python scripts/bake_tissue_table.py

The notebook has to build input channels - perfusion, rho*c, k, sigma - to draw
an anatomy or to run the interactive planner, and those numbers come out of
the solver's material database. Requiring a compiled C++ solver on Colab for five
constants is not a trade anyone should make, so the constants are baked, with
the database's own revision hash. `device.verify_tissue_table()` compares the
two wherever the solver is present, so drift is caught rather than assumed away.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d.device import (TISSUE_TABLE_PATH, query_solver_tissue,  # noqa: E402
                               verify_tissue_table)

if __name__ == "__main__":
    table = query_solver_tissue()
    TISSUE_TABLE_PATH.write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote {TISSUE_TABLE_PATH}")
    print(f"  solver version     {table['_solver_version']}")
    print(f"  material revision {table['_material_revision']}")
    for k, v in table["tissues"].items():
        print(f"  {k:11s} <- {v['db_name']:19s} rho*c {v['rho'] * v['c']:.4e}  "
              f"k {v['k']:.4e}  sigma {v['sigma']:.4e}  perf {v['perfusion']:.4e}")
    print(verify_tissue_table())
