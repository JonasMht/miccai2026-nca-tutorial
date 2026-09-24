"""Training, with the tricks an NCA needs and a CNN does not.

Each trick is marked `# TRICK n` in the code below and matches the talk:

1. Sample the rollout length K every batch. A fixed K teaches a sequence of
   moves that ends at step K; a random K forces a rule that settles.
2. Sometimes start from a state the model reaches on its own (rolled forward
   without gradient first), not only from the blank seed.
3. Normalise each parameter's gradient by its own norm, then clip globally as
   a second safety net.
4. Persistence: continue the same rollout for 40-200 more steps and supervise
   it again, with gradient on the last few steps only (truncated BPTT). This is
   what keeps the field stable when you keep iterating.
5. The rollout length at inference is chosen on validation after training:
   the cheapest K within 1 % of the best.
"""
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .data import AblationDataset
from .model import (AblationCNCA, best_vessel_threshold, dice,  # noqa: F401
                    multitask_loss, vessel_f1)


@dataclass
class TrainConfig:
    # architecture: 16 channels = 2 answers + 3 environment + 11 scratch.
    # Width lifted both heads together (6 ch: DSC 0.30, 16 ch: 0.71); 24 buys little.
    channels: int = 16
    hidden_mult: int = 3
    n_sub_models: int = 3
    fire_rate: float = 1.0          # every cell updates every step; 0.5 measured worse
    restore_conditioning: bool = True

    # rollout
    step_min: int = 3
    step_max: int = 18
    infer_steps: int = 15

    # optimisation
    epochs: int = 300
    batch_size: int = 16
    lr: float = 2e-3
    lr_min: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.0

    # robustness to long rollouts (TRICK 2 and 4). The persistence range has to
    # cover the horizon you care about: with (40, 200) the field is flat out to
    # 400 steps. A long range sampled rarely is cheaper than a short one sampled
    # often, because a no-grad step costs nearly as much as a gradient step.
    further_steps_rate: float = 0.07
    persist_range: tuple = (40, 200)   # extra steps, supervised a second time
    persist_bptt: int = 8              # how many of them carry a gradient
    persist_rate: float = 0.1
    # The vessels do not depend on the plan, so for this fraction of each batch
    # the needle and power channels are taken from another sample and only the
    # vessel head is trained. It keeps the vessel map from following the needle.
    plan_shuffle_rate: float = 0.0
    # Exponential moving average of the weights, used for validation and kept
    # at the end. 0 turns it off.
    ema_decay: float = 0.0
    # Extra vessel-loss weight inside and around the predicted lesion.
    lesion_boost: float = 0.0
    # Steps run on the CT alone before the needle is switched on; the vessel
    # channel is then held fixed (see AblationCNCA). 0 is the single-phase model.
    anatomy_steps: int = 0
    # If the validation loss rises above this multiple of its best (or anything
    # goes non-finite), training goes back to the best weights with a fresh
    # optimiser and half the learning rate. 0 turns the guard off.
    diverge_factor: float = 2.0

    # loss
    foreground_weight: float = 100.0
    # Weight of the vessel term. Check the share it gives each head, not the
    # number: at 3.0 the necrosis head received 3 % of the loss, and 0.3 moved
    # validation DSC at epoch 10 from 0.46 to 0.79.
    vessel_weight: float = 0.3
    vessel_pos_weight: float = 20.0

    # Rotations and flips of each sample. Off: axial CT has an orientation, and
    # training with it cost the vessel head most of its F1 (0.70 -> 0.28).
    augment_d4: bool = False

    # Stop between epochs when this many seconds have passed. The LR schedule
    # follows the clock too, so a slow GPU still ends on a small learning rate.
    time_budget_s: float | None = None

    # runtime
    device: str = "cuda"
    amp_dtype: str = "auto"      # "auto" | "fp32"; see _amp
    seed: int = 0
    validate_every: int = 5
    grad_norm_eps: float = 1e-8


#: The session recipe for a Colab T4: 12 channels (7 scratch) is the narrowest
#: width at which both heads clearly work.
FAST = TrainConfig(channels=12, hidden_mult=2, n_sub_models=2,
                   step_min=3, step_max=10, infer_steps=8,
                   epochs=40, batch_size=16, lr=3e-3,
                   persist_range=(40, 200), persist_bptt=6)

#: The reference model the notebook loads.
FULL = TrainConfig()


# --------------------------------------------------------------------------- #
# the dihedral group of the square, for augmentation and test-time averaging
# --------------------------------------------------------------------------- #
N_D4 = 8


def d4(t: torch.Tensor, g: int) -> torch.Tensor:
    """Element `g` of D4 on the last two axes: `g % 4` quarter turns, then a
    mirror if `g >= 4`."""
    out = torch.rot90(t, g % 4, dims=(-2, -1))
    if g >= 4:
        out = torch.flip(out, dims=(-1,))
    return out.contiguous()      # rot90/flip return strided views; convs prefer contiguous


def d4_inv(t: torch.Tensor, g: int) -> torch.Tensor:
    """Inverse of `d4(., g)`: for a mirrored element, un-mirror first, then rotate back."""
    if g >= 4:
        return torch.rot90(torch.flip(t, dims=(-1,)), -(g % 4), dims=(-2, -1)).contiguous()
    return torch.rot90(t, -(g % 4), dims=(-2, -1)).contiguous()


@torch.no_grad()
def predict_repeats(model, inputs: torch.Tensor, steps: int, n: int = 8) -> torch.Tensor:
    """Mean of `n` rollouts. Firing is random, so a single rollout is one sample."""
    return sum(model(inputs, steps=steps) for _ in range(n)) / n


@torch.no_grad()
def predict_d4_tta(model, inputs: torch.Tensor, steps: int, *,
                   reduce: str = "mean") -> torch.Tensor:
    """Mean over the eight orientations, each mapped back before averaging.

    Most of the gain over one rollout is the averaging of stochastic draws, not
    the orientations; compare with `predict_repeats`.
    """
    acc = None
    for g in range(N_D4):
        p = model(d4(inputs, g) if g else inputs, steps=steps)
        p = d4_inv(p, g) if g else p
        acc = p if acc is None else acc + p
    return acc / N_D4 if reduce == "mean" else acc


def augment_d4(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator):
    """Re-orient every sample of a batch independently, uniformly over D4.

    Uses its own generator so that switching augmentation on does not change
    the rollout lengths drawn from the global stream.
    """
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(f"D4 augmentation needs a square grid, got {tuple(x.shape[-2:])}")
    g = torch.randint(0, N_D4, (x.shape[0],), generator=gen, device=x.device)
    xo, yo = torch.empty_like(x), torch.empty_like(y)
    for k in range(N_D4):
        m = g == k
        if not bool(m.any()):
            continue
        if k == 0:
            xo[m], yo[m] = x[m], y[m]
        else:
            xo[m], yo[m] = d4(x[m], k), d4(y[m], k)
    return xo, yo


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_GPU_PROBE_OK = True


def _gpu_state() -> str:
    """SM clock, max clock, temperature and throttle flags, for the training log.
    A duration measured on a throttled GPU is not comparable to one that is not."""
    global _GPU_PROBE_OK
    if not _GPU_PROBE_OK or not torch.cuda.is_available():
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
             "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        sm, mx, tc, reasons = [f.strip() for f in out.splitlines()[0].split(",")]
        flags = int(reasons, 16)
        names = [n for bit, n in ((0x4, "PWRCAP"), (0x8, "HWSLOW"), (0x20, "THERMAL"),
                                  (0x40, "HWTHERMAL"), (0x80, "PWRBRAKE")) if flags & bit]
        return f"  [{sm}/{mx}MHz {tc}C {'/'.join(names) or 'clear'}]"
    except Exception:
        _GPU_PROBE_OK = False
        return ""


def _amp(cfg):
    """Mixed precision: bf16 on Ampere and newer, fp16 on older cards.

    `torch.cuda.is_bf16_supported()` says yes on a T4 or a V100 because bf16
    can be emulated there, and emulated it is ~30x slower than fp32 (measured
    on a TITAN V: 1277 ms against 39 ms per batch). So ask for the compute
    capability instead. fp16 needs a GradScaler; `train` creates one.
    """
    if not cfg.device.startswith("cuda") or cfg.amp_dtype == "fp32":
        return False, torch.float32
    major, _ = torch.cuda.get_device_capability(cfg.device)
    return True, (torch.bfloat16 if major >= 8 else torch.float16)


def normalise_grads(model, eps: float):
    """TRICK 3: divide each parameter's gradient by its own norm."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + eps)


@torch.no_grad()
def evaluate(model, ds: AblationDataset, steps: int, batch_size: int = 16,
             foreground_weight: float = 100.0, vessel_weight: float = 1.0):
    """(loss, necrosis DSC)."""
    return evaluate_full(model, ds, steps, batch_size, foreground_weight, vessel_weight)[:2]


@torch.no_grad()
def evaluate_full(model, ds: AblationDataset, steps: int, batch_size: int = 16,
                  foreground_weight: float = 100.0, vessel_weight: float = 1.0,
                  tune_threshold: bool = True):
    """(loss, necrosis DSC, vessel F1).

    The vessel F1 is taken at the best threshold on this split; the threshold
    is left in `evaluate_full.vessel_threshold`.
    """
    model.eval()
    preds, tgts = [], []
    tot_loss = tot_dice = 0.0
    n = 0
    for x, y in ds.batches(batch_size, shuffle=False):
        p = model(x, steps=steps)
        tot_loss += float(multitask_loss(p, y, foreground_weight, vessel_weight)) * x.shape[0]
        tot_dice += float(dice(p, y).sum())
        n += x.shape[0]
        if y.shape[1] > 1:
            preds.append(p.detach())
            tgts.append(y)
    model.train()
    f1 = thr = 0.0
    if preds:
        P, T = torch.cat(preds), torch.cat(tgts)
        if tune_threshold:
            thr, f1 = best_vessel_threshold(P, T)
        else:
            thr, f1 = 0.5, float(vessel_f1(P, T).mean())
    evaluate_full.vessel_threshold = thr
    return tot_loss / n, tot_dice / n, f1


@torch.no_grad()
def choose_inference_steps(model, ds: AblationDataset, candidates=range(4, 41, 2),
                           batch_size: int = 16, tolerance: float = 0.01,
                           on: str = "dice", return_vessel: bool = False):
    """TRICK 5: sweep K on validation, return the cheapest within `tolerance` of the best.

    Selects on the necrosis DSC by default. `return_vessel=True` also returns
    the vessel F1 curve, since the two heads need not want the same K.
    """
    scores, vscores = {}, {}
    have_vessel = getattr(ds, "targets", None) is not None and ds.targets.shape[1] > 1
    for k in map(int, candidates):
        scores[k] = evaluate(model, ds, steps=k, batch_size=batch_size)[1]
        if have_vessel and (return_vessel or on == "vessel_f1"):
            pv = torch.cat([model(x, steps=k) for x, _ in ds.batches(batch_size, shuffle=False)])
            vscores[k] = best_vessel_threshold(pv, ds.targets)[1]
    pick = vscores if on == "vessel_f1" and vscores else scores
    best = max(pick.values())
    cheapest = min(k for k, v in pick.items() if v >= best * (1.0 - tolerance))
    if return_vessel:
        return cheapest, scores, vscores
    return cheapest, scores


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, train_ds: AblationDataset, val_ds: AblationDataset,
          *, model=None, out_dir: Path | None = None, on_epoch=None, log=print,
          select: str = "dice"):
    """Train and return `(model, history)` with the best validation weights loaded.

    `model=` trains a model you built yourself (the notebook passes its own
    class); otherwise one is built from `cfg`. `on_epoch(epoch, record, model)`
    is called after every epoch. `select` picks the kept epoch: "dice",
    "vessel_f1" or "both" (their sum).
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    if model is None:
        model = AblationCNCA(channels=cfg.channels, hidden_mult=cfg.hidden_mult,
                             n_sub_models=cfg.n_sub_models, fire_rate=cfg.fire_rate,
                             restore_conditioning=cfg.restore_conditioning,
                             anatomy_steps=cfg.anatomy_steps)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp, amp_dtype = _amp(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    gen = torch.Generator(device=train_ds.torch_device).manual_seed(cfg.seed)
    aug_gen = (torch.Generator(device=train_ds.torch_device).manual_seed(cfg.seed ^ 0x5F11)
               if cfg.augment_d4 else None)

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

    def sample_k():
        return int(torch.randint(cfg.step_min, cfg.step_max + 1, (1,)).item())

    ema = [p.detach().clone() for p in model.parameters()] if cfg.ema_decay else None

    def swap_ema():
        """Exchange the live weights and the averaged ones (a no-op without EMA)."""
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema, model.parameters()):
                    tmp = p.detach().clone()
                    p.copy_(e)
                    e.copy_(tmp)

    history: list[dict] = []
    best = {"score": -1.0, "epoch": -1}
    lr_scale, best_vl, anchor, anchor_live = 1.0, math.inf, None, None
    t0 = time.time()
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)

    for epoch in range(cfg.epochs):
        elapsed = time.time() - t0
        epoch_s = elapsed / epoch if epoch else 0.0
        if cfg.time_budget_s is not None and epoch and elapsed + epoch_s > cfg.time_budget_s:
            log(f"  time budget {cfg.time_budget_s:.0f}s: stopping after {epoch} epochs "
                f"({elapsed:.0f}s), keeping the best validation weights (epoch {best['epoch']})")
            if history:
                history[-1]["stopped_for_time"] = True
            break
        # the last epoch the budget allows is always validated
        last = epoch == cfg.epochs - 1 or (cfg.time_budget_s is not None and epoch
                                           and elapsed + 2 * epoch_s > cfg.time_budget_s)

        # Warmup, then cosine. Under a time budget the cosine follows the clock
        # as well as the epoch count.
        if epoch < cfg.warmup_epochs:
            lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
        else:
            t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            if cfg.time_budget_s is not None:
                t = max(t, (elapsed + epoch_s) / cfg.time_budget_s)
            t = min(t, 1.0)
            lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * t))
        lr *= lr_scale
        for g in opt.param_groups:
            g["lr"] = lr

        run_loss = 0.0
        for x, y in train_ds.batches(cfg.batch_size, generator=gen):
            if aug_gen is not None:
                x, y = augment_d4(x, y, aug_gen)
            n_shuf = int(round(cfg.plan_shuffle_rate * x.shape[0]))
            if n_shuf:
                x = x.clone()
                x[:n_shuf, 1:] = x[:n_shuf, 1:].roll(1, dims=0) if n_shuf > 1 \
                    else x[-1:, 1:]
            state = None                   # the model seeds itself (and reads the anatomy first)

            # TRICK 2: sometimes start from a state the model reaches on its own
            if cfg.further_steps_rate > 0 and torch.rand(1).item() < cfg.further_steps_rate:
                with torch.no_grad():
                    _, state = model(x, steps=sample_k(), return_state=True)
                state = state.detach()

            k = sample_k()                                     # TRICK 1
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred, rolled = model(x, steps=k, state=state, return_state=True)
            loss = multitask_loss(pred.float(), y, cfg.foreground_weight,
                                  cfg.vessel_weight, cfg.vessel_pos_weight, n_shuf,
                                  cfg.lesion_boost)

            # TRICK 4: persistence. Keep rolling the same state without gradient,
            # then supervise the last `persist_bptt` steps again.
            if cfg.persist_rate > 0 and torch.rand(1).item() < cfg.persist_rate:
                extra = int(torch.randint(*cfg.persist_range, (1,)).item())
                free = max(0, extra - cfg.persist_bptt)
                s2 = rolled.detach()
                if free:
                    with torch.no_grad(), autocast():
                        _, s2 = model(x, steps=free, state=s2, return_state=True)
                    s2 = s2.detach()
                with autocast():
                    pred2 = model(x, steps=min(extra, cfg.persist_bptt), state=s2)
                loss = loss + multitask_loss(pred2.float(), y, cfg.foreground_weight,
                                             cfg.vessel_weight, cfg.vessel_pos_weight, n_shuf,
                                  cfg.lesion_boost)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            normalise_grads(model, cfg.grad_norm_eps)                # TRICK 3
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                with torch.no_grad():
                    for e, p in zip(ema, model.parameters()):
                        e.lerp_(p.detach(), 1.0 - cfg.ema_decay)
            run_loss += float(loss.detach())

        rec = {"epoch": epoch, "lr": lr, "train_loss": run_loss / steps_per_epoch,
               "wall_s": round(time.time() - t0, 1)}
        if epoch % cfg.validate_every == 0 or last:
            rec["gpu"] = _gpu_state()
            swap_ema()
            vl, vd, vf = evaluate_full(model, val_ds, cfg.infer_steps, cfg.batch_size,
                                       cfg.foreground_weight, cfg.vessel_weight)
            rec.update(val_loss=vl, val_dice=vd, val_vessel_f1=vf,
                       vessel_threshold=evaluate_full.vessel_threshold)
            # the weights in `model` here are the evaluated ones (the average,
            # with EMA on); the live ones sit in `ema` until swap_ema() below
            if math.isfinite(vl) and vl < best_vl:
                best_vl = vl
                anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
                anchor_live = [e.detach().clone() for e in ema] if ema is not None else None
            elif cfg.diverge_factor and anchor is not None and (
                    not math.isfinite(vl) or not math.isfinite(rec["train_loss"])
                    or vl > cfg.diverge_factor * best_vl):
                model.load_state_dict(anchor)
                if ema is not None:
                    for e, a in zip(ema, anchor_live):
                        e.copy_(a)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
                lr_scale *= 0.5
                rec["rolled_back"] = True
                log(f"  epoch {epoch}: validation loss {vl:.3f} against a best of {best_vl:.3f}, "
                    f"back to the best weights, learning rate x{lr_scale:g}")
                vl, vd, vf = best_vl, -1.0, -1.0
            # Validation DSC is noisy from one pass to the next (about +-0.04),
            # so the kept epoch is partly luck; "both" smooths that a little.
            score = {"vessel_f1": vf, "both": vd + vf}.get(select, vd)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
        history.append(rec)
        if on_epoch is not None:
            on_epoch(epoch, rec, model)
        elif "val_dice" in rec:
            log(f"  epoch {epoch:4d}  loss {rec['train_loss']:.5f}  val {rec['val_loss']:.5f}  "
                f"DSC {rec['val_dice']:.4f}  vesselF1 {rec['val_vessel_f1']:.4f}"
                f"@{rec['vessel_threshold']:.2f}  {rec['wall_s']:.0f}s{rec['gpu']}")
        if "val_dice" in rec:
            swap_ema()

    if best["epoch"] >= 0:
        model.load_state_dict(best["state"])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                    "best_epoch": best["epoch"], "best_val_dice": best["score"],
                    "vessel_threshold": getattr(evaluate_full, "vessel_threshold", 0.5)},
                   out_dir / "model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return model, history


def load_checkpoint(path, device="cpu"):
    """(model, cfg, checkpoint).

    Config keys the checkpoint does not have take today's defaults and are
    listed in `ck["config_defaulted"]`; keys that no longer exist are ignored.
    """
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    stored = dict(ck["config"])
    ck["config_defaulted"] = sorted(known - set(stored))
    ck["config_unknown"] = sorted(set(stored) - known)
    cfg = TrainConfig(**{k: v for k, v in stored.items() if k in known})
    model = AblationCNCA(channels=cfg.channels, hidden_mult=cfg.hidden_mult,
                         n_sub_models=cfg.n_sub_models, fire_rate=cfg.fire_rate,
                         restore_conditioning=cfg.restore_conditioning,
                         anatomy_steps=cfg.anatomy_steps)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, cfg, ck
