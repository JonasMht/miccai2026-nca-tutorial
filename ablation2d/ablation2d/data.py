"""The corpus, held resident on the device as uint8 and dequantised per batch.

The whole corpus fits in memory, so there is no DataLoader and no disk I/O
during training. Keeping it as uint8 on the GPU costs 4x less memory than
float32, which on a small GPU allows a 4x larger batch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .channels import CHANNEL_NAMES, N_INPUT_CHANNELS


def needle_counts(inputs: np.ndarray) -> np.ndarray:
    """Needles per plan, from the stored uint8 inputs.

    Each needle has its own dial setting, so the number of distinct nonzero
    power levels at the radiating slots is the number of needles. (Two needles
    at exactly the same power would count as one; the corpus draws powers from
    a continuous distribution.)
    """
    act = inputs[:, 2]
    return np.array([len(np.unique(a[a > 0])) for a in act])


class AblationDataset:
    """One split on `device`: `inputs` (N, 3, H, W) and `targets` (N, 2, H, W), in [0, 1]."""

    def __init__(self, path: str | Path, device: str | torch.device = "cpu",
                 limit: int | None = None, max_needles: int | None = None):
        z = np.load(Path(path))
        x = z["inputs"]
        y = z["targets"]
        if max_needles is not None:
            keep = needle_counts(x) <= max_needles
            x, y = x[keep], y[keep]
        if limit is not None:
            x, y = x[:limit], y[:limit]
        # _deq divides in place, which is only safe on a fresh float copy of uint8 data
        if x.dtype != np.uint8 or y.dtype != np.uint8:
            raise TypeError(f"corpus must be uint8, got {x.dtype}/{y.dtype}")
        self._x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self._y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        self.device = device
        if self._x.shape[1] != N_INPUT_CHANNELS:
            raise ValueError(f"corpus has {self._x.shape[1]} channels, "
                             f"model expects {N_INPUT_CHANNELS} ({CHANNEL_NAMES})")

    @staticmethod
    def _deq(t):
        return t.float().div_(255.0)

    @property
    def torch_device(self) -> torch.device:
        """The device the tensors live on."""
        return self._x.device

    @property
    def inputs(self) -> torch.Tensor:
        """The whole split as float32 (a copy; prefer `sample` or `batches`)."""
        return self._deq(self._x)

    @property
    def targets(self) -> torch.Tensor:
        """(N, 2, H, W): necrosis, then vessels."""
        return self._deq(self._y)

    @property
    def cell_death(self) -> torch.Tensor:
        """(N, 1, H, W): necrosis only."""
        return self._deq(self._y[:, 0:1])

    def __len__(self) -> int:
        return self._x.shape[0]

    @property
    def shape(self):
        return tuple(self._x.shape[2:])

    def sample(self, idx):
        """One or more cases as float32, batched."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self._deq(self._x[idx]), self._deq(self._y[idx])

    def batches(self, batch_size: int, generator: torch.Generator | None = None,
                shuffle: bool = True):
        n = len(self)
        order = torch.randperm(n, generator=generator, device=self._x.device) \
            if shuffle else torch.arange(n, device=self._x.device)
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            yield self._deq(self._x[sel]), self._deq(self._y[sel])

    def stats(self) -> dict:
        frac = (self._y[:, 0] >= 252).flatten(1).float().mean(1)   # 0.99 * 255
        ves = (self._y[:, 1] > 127).flatten(1).float().mean(1) if self._y.shape[1] > 1 \
            else None
        out = {
            "n": len(self),
            "grid": self.shape,
            "lesion_frac_pct": [round(float(frac.min()) * 100, 2),
                                round(float(frac.median()) * 100, 2),
                                round(float(frac.max()) * 100, 2)],
            "channel_max": {n: round(float(self._x[:, i].max()) / 255.0, 3)
                            for i, n in enumerate(CHANNEL_NAMES)},
        }
        if ves is not None:
            out["vessel_frac_pct"] = round(float(ves.mean()) * 100, 2)
        return out


def load_splits(root: str | Path, device="cpu", limit=None, max_needles=None):
    root = Path(root)
    return {s: AblationDataset(root / f"{s}.npz", device, limit, max_needles)
            for s in ("train", "val", "test") if (root / f"{s}.npz").exists()}
