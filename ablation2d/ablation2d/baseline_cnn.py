"""A parameter-matched dilated CNN: the control for "what does iterating buy?".

Same inputs, corpus, loss, split and schedule as the NCA, about the same
parameter count, and a larger receptive field: dilations 1-32 on 3x3 kernels
reach 127 cells against the NCA's 90. No downsampling, so the one variable is
the recurrence.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .channels import N_INPUT_CHANNELS

DILATIONS = (1, 2, 4, 8, 16, 32)


class DilatedCNN(nn.Module):
    def __init__(self, width: int = 28, dilations=DILATIONS,
                 in_channels: int = N_INPUT_CHANNELS, out_channels: int = 2):
        super().__init__()
        self.dilations = tuple(dilations)
        layers = [nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU()]
        for d in self.dilations:
            layers += [nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.ReLU()]
        layers += [nn.Conv2d(width, out_channels, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def seed(self, inputs):
        """No state to seed; returns the inputs so the NCA training loop runs unchanged."""
        return inputs

    def forward(self, inputs, steps=None, state=None,
                return_trace=False, return_state=False):
        """`steps` is ignored; the signature matches the NCA so both share one evaluation."""
        out = torch.sigmoid(self.net(inputs))
        if return_trace:
            return (out, [out], inputs) if return_state else (out, [out])
        return (out, inputs) if return_state else out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def reach_cells(self) -> int:
        return 1 + 2 * sum(self.dilations)


def match_params(target: int, dilations=DILATIONS, tol: float = 0.10) -> int:
    """The width whose parameter count is closest to `target` (within `tol`)."""
    best, best_err = None, float("inf")
    for w in range(4, 129):
        n = DilatedCNN(w, dilations).n_params
        err = abs(n - target) / target
        if err < best_err:
            best, best_err = w, err
    if best_err > tol:
        raise RuntimeError(f"closest width {best} is {best_err:.1%} off {target}")
    return best
