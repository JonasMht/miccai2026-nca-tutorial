"""Import the solver lazily, from its source checkout.

Only the corpus generator needs the solver. Training, evaluation and the
planners run on the `.npz` corpus and a baked tissue table, so importing the
package must never require it (Colab has no solver). Hence `require_solver()`
instead of a module-level import.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_CANDIDATES = [
    os.environ.get("SOLVER_PYTHON"),
    "/home/jonas/Documents/Research/UniPhys/python",
    str(Path.home() / "Documents/Research/UniPhys/python"),
]

#: Bindings added for this workshop; without them the device is uncalibrated and
#: the vessels thermally inert, so refuse to run.
_REQUIRED = ["set_mw_calibration", "_set_vessel_radius"]


@lru_cache(maxsize=1)
def require_solver():
    """The solver module, or a clear ImportError explaining what needs it."""
    mod = None
    try:
        import uniphys as mod  # noqa: F401
    except ImportError:
        for c in _CANDIDATES:
            if c and (Path(c) / "uniphys" / "__init__.py").exists():
                sys.path.insert(0, c)
                try:
                    import uniphys as mod  # noqa: F811
                except ImportError:
                    mod = None
                break
    if mod is None:
        raise ImportError(
            "the solver is needed for this operation (solving new ground truth). "
            "Set SOLVER_PYTHON to the repo's python/ directory, or install the "
            "wheel. Training, evaluation and the planners do not need it."
        )
    missing = [n for n in _REQUIRED
               if not (hasattr(mod, n) or hasattr(mod.Simulation, n))]
    if missing:
        raise ImportError(
            f"the solver is too old for this workshop: missing {', '.join(missing)}. "
            "Rebuild the python module (cmake --build build/py --target _uniphys)."
        )
    return mod


def solver_available() -> bool:
    try:
        require_solver()
        return True
    except ImportError:
        return False
