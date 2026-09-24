"""A 2-D chained NCA for microwave ablation: necrosis and vessels from one rollout.

The whole grid is one tensor, and its channels have fixed roles:

    [0:2]    the answers: necrosis, vessels (read out through a sigmoid)
    [2:5]    the environment: CT, needle, power
    [5:C]    scratch channels

One CA step runs N small blocks in sequence, each adding a residual update
("chained"). Each block is two 3x3 convolutions, so a step reaches 2N cells.

There is no normalisation layer: it would read statistics over the whole grid,
and a cell that sees the whole grid is no longer a cellular automaton. The state
is kept bounded locally instead, by a soft clamp, a small init and a zero-init
output layer.

With `restore_conditioning=True` (the default) the environment is rewritten
after every step. Without it the automaton may overwrite its own inputs, and it
does: the 3-D surrogate of the training codebase, which does not restore,
drifts its sparse power channel to 7630x its own mean within nine steps.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .channels import (CHANNEL_NAMES, N_INPUT_CHANNELS,
                       N_TARGET_CHANNELS, TARGET_NAMES)

OUT_SLICE = slice(0, N_TARGET_CHANNELS)
COND_SLICE = slice(N_TARGET_CHANNELS, N_TARGET_CHANNELS + N_INPUT_CHANNELS)

#: Bound on the state, applied as `c * tanh(x / c)`. A hard clamp has zero
#: gradient outside its bounds and training stalls once the state reaches them.
STATE_CLAMP = 4.0


def channel_roles(n_channels: int, restored: bool = True) -> list[str]:
    """A name for every channel of the state, in order."""
    kind = "cond" if restored else "was"
    names = list(TARGET_NAMES) + [f"{kind}: {c}" for c in CHANNEL_NAMES]
    return names + [f"hidden {i}" for i in range(n_channels - len(names))]


class AblationCNCA(nn.Module):
    """Chained NCA predicting the necrosis field and segmenting the vessels.

    Args:
        channels: state width: 2 answers + 3 environment + scratch.
        hidden_mult: block width, as a multiple of `channels`.
        n_sub_models: chained blocks per CA step.
        fire_rate: probability that a cell applies its update on a sub-step.
            1 (every cell, every step, deterministic) is the default; the
            classic 0.5 measured worse here and is kept only to load older
            checkpoints.
        restore_conditioning: rewrite the environment channels every step.
        anatomy_steps: if > 0, a rollout from the seed first runs this many
            steps with the needle and power channels empty, so the vessels are
            found from the CT alone, and then keeps the vessel channel fixed
            while the planned steps run. The vessel answer cannot depend on the
            plan. 0 is the single-phase model.
    """

    def __init__(self, channels: int = 16, hidden_mult: int = 3,
                 n_sub_models: int = 3, kernel_size: int = 3,
                 fire_rate: float = 1.0, restore_conditioning: bool = True,
                 anatomy_steps: int = 0):
        super().__init__()
        if channels < N_TARGET_CHANNELS + N_INPUT_CHANNELS:
            raise ValueError(f"channels={channels} cannot hold {N_TARGET_CHANNELS} answers "
                             f"and {N_INPUT_CHANNELS} input channels")
        self.channels = channels
        self.n_sub_models = n_sub_models
        self.fire_rate = fire_rate
        self.restore_conditioning = restore_conditioning
        self.anatomy_steps = anatomy_steps
        pad = kernel_size // 2
        hidden = channels * hidden_mult
        self.sub_models = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, kernel_size, padding=pad, bias=False),
            ) for _ in range(n_sub_models)
        ])
        for sub in self.sub_models:
            first, last = sub[0], sub[2]
            # He init scaled down 10x: the update is composed with itself tens of times
            fan_in = first.weight[0].numel()
            nn.init.normal_(first.weight, 0.0, (2.0 / fan_in) ** 0.5 * 0.1)
            nn.init.zeros_(first.bias)
            # every block starts as a no-op, so the untrained output is exactly flat
            nn.init.zeros_(last.weight)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        for sub in self.sub_models:
            dx = sub(x)
            if self.fire_rate < 1.0:
                # one draw per cell, shared by all its channels
                fire = (torch.rand_like(x[:, :1]) <= self.fire_rate).to(x.dtype)
                dx = dx * fire
            x = STATE_CLAMP * torch.tanh((x + dx) / STATE_CLAMP)
        return x

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        """State -> (B, 2, H, W) answers in [0, 1]."""
        return torch.sigmoid(x[:, OUT_SLICE])

    def seed(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) inputs -> the initial state: zeros plus the environment."""
        b, _, h, w = inputs.shape
        x = inputs.new_zeros(b, self.channels, h, w)
        x[:, COND_SLICE] = inputs
        return x

    def forward(self, inputs: torch.Tensor, steps: int = 15, *,
                state: torch.Tensor | None = None,
                return_trace: bool = False, return_state: bool = False,
                trace_full: bool = False):
        """(B, 2, H, W) in [0, 1]: necrosis, then vessels.

        `state` continues a previous rollout instead of starting from the seed.
        `return_trace` also returns one frame per step: the answers, or with
        `trace_full=True` the whole raw state (answers as logits).
        """
        if state is None:
            x = self.seed(inputs)
            if self.anatomy_steps:
                anatomy = inputs.clone()
                anatomy[:, 1:] = 0                     # no needle yet: read the CT only
                x = self.seed(anatomy)
                for _ in range(self.anatomy_steps):
                    x = self.step(x)
                    x = torch.cat([x[:, OUT_SLICE], anatomy, x[:, COND_SLICE.stop:]], dim=1)
        else:
            x = state
        vessels = x[:, 1:2] if self.anatomy_steps else None
        trace = []
        for _ in range(steps):
            x = self.step(x)
            if self.restore_conditioning:
                x = torch.cat([x[:, OUT_SLICE], inputs, x[:, COND_SLICE.stop:]], dim=1)
            if vessels is not None:                    # found once, then held fixed
                x = torch.cat([x[:, :1], vessels, x[:, 2:]], dim=1)
            if return_trace:
                trace.append((x if trace_full else self.readout(x)).detach().clone())
        out = self.readout(x)
        if return_trace:
            return (out, trace, x) if return_state else (out, trace)
        return (out, x) if return_state else out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def reach_cells(self) -> int:
        """Receptive field gained per CA step: 2 cells per block."""
        return 2 * self.n_sub_models


# --------------------------------------------------------------------------- #
# losses and metrics
# --------------------------------------------------------------------------- #
def necrosis_loss(pred: torch.Tensor, target: torch.Tensor,
                  foreground_weight: float = 100.0,
                  background_weight: float = 1.0,
                  threshold: float = 0.01) -> torch.Tensor:
    """Foreground-weighted MSE.

    The lesion covers 1-4 % of the slice, so plain MSE is minimised by
    predicting nothing. At 100:1 a missed necrotic cell costs as much as a
    hundred invented ones, which biases the model towards over-predicting the
    zone: the safer error for an ablation margin.
    """
    fg = torch.as_tensor(foreground_weight, device=pred.device, dtype=pred.dtype)
    bg = torch.as_tensor(background_weight, device=pred.device, dtype=pred.dtype)
    w = torch.where(target > threshold, fg, bg)
    return (w * (pred - target) ** 2).sum() / w.sum()


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, averaged over the batch. Insensitive to the class ratio,
    which matters for a mask covering ~3 % of the slice."""
    p = pred.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def masked_bce(pred: torch.Tensor, target: torch.Tensor,
               pos_weight: float = 20.0, eps: float = 1e-6,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Positive-weighted BCE on probabilities. Gives every cell a gradient from
    the first step, where Dice alone can sit in a degenerate solution."""
    p = pred.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * target * torch.log(p) + (1 - target) * torch.log(1 - p))
    return (loss if weight is None else loss * weight).mean()


def multitask_loss(pred: torch.Tensor, target: torch.Tensor,
                   death_weight: float = 100.0, vessel_weight: float = 1.0,
                   vessel_pos_weight: float = 20.0, vessel_only: int = 0,
                   lesion_boost: float = 0.0) -> torch.Tensor:
    """Necrosis (weighted MSE) + vessel_weight * vessels (Dice + weighted BCE).

    The first `vessel_only` samples of the batch count for the vessel term only:
    their necrosis target is unknown (see `plan_shuffle_rate` in train.py).

    `lesion_boost` multiplies the vessel BCE by 1 + lesion_boost where the model
    currently predicts necrosis (and 4 mm around it). Without it the model learns
    that vessels are rare inside a lesion and stops seeing them there.
    """
    w = None
    if lesion_boost:
        zone = torch.nn.functional.max_pool2d((pred[:, 0:1].detach() > 0.5).float(), 5, 1, 2)
        w = 1.0 + lesion_boost * zone
    v = (soft_dice_loss(pred[:, 1:2], target[:, 1:2])
         + masked_bce(pred[:, 1:2], target[:, 1:2], vessel_pos_weight, weight=w))
    n = vessel_only
    return necrosis_loss(pred[n:, 0:1], target[n:, 0:1], death_weight) + vessel_weight * v


@torch.no_grad()
def vessel_f1(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5) -> torch.Tensor:
    """Per-sample F1 of the vessel head (channel 1)."""
    p = (pred[:, 1] > threshold).flatten(1).float()
    t = (target[:, 1] > 0.5).flatten(1).float()
    tp = (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    return torch.where(den > 0, 2 * tp / den, torch.ones_like(den))


@torch.no_grad()
def best_vessel_threshold(pred: torch.Tensor, target: torch.Tensor,
                          candidates=None):
    """The vessel decision threshold with the best F1; choose it on validation.

    The vessel loss weights positives 20:1, which makes the head over-confident,
    so the best threshold is well above 0.5. The grid goes to 0.95 so that the
    optimum is never pinned at its edge. Returns `(threshold, f1)`.
    """
    cands = candidates if candidates is not None else \
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98]
    best = (0.5, -1.0)
    for t in cands:
        f = float(vessel_f1(pred, target, t).mean())
        if f > best[1]:
            best = (float(t), f)
    return best


@torch.no_grad()
def dice(pred: torch.Tensor, target: torch.Tensor, contour: float = 0.99,
         pred_threshold: float = 0.5) -> torch.Tensor:
    """Per-sample DSC of the necrosis (channel 0).

    The target is cut at the Arrhenius d >= 0.99 contour, where the tissue is
    dead; the prediction at 0.5.
    """
    if pred.shape[1] > 1:
        pred, target = pred[:, 0:1], target[:, 0:1]
    p = (pred > pred_threshold).flatten(1).float()
    t = (target >= contour).flatten(1).float()
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    return torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
