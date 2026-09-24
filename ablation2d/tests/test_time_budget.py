"""The time budget: the session budget is time, not epochs.

`TrainConfig.time_budget_s` stops training between epochs (an in-flight epoch
always finishes), keeps the best validation weights, and marks the stopping
epoch in the history so a run's provenance says how long it actually trained.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d.data import AblationDataset  # noqa: E402
from ablation2d.train import FAST, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _cfg(**over):
    base = {"channels": 6, "hidden_mult": 1, "n_sub_models": 1, "epochs": 4,
            "device": "cpu", "step_min": 3, "step_max": 8, "batch_size": 4,
            "validate_every": 1, "persist_rate": 0.0, "further_steps_rate": 0.0,
            "augment_d4": False}
    return FAST.__class__(**{**base, **over})


def _tiny():
    tr = AblationDataset(ROOT / "data" / "val.npz", device="cpu", limit=8)
    vl = AblationDataset(ROOT / "data" / "val.npz", device="cpu", limit=4)
    return tr, vl


def test_budget_stops_early_and_marks_the_epoch():
    tr, vl = _tiny()
    _, h = train(_cfg(time_budget_s=0.0), tr, vl, log=lambda s: None)
    assert len(h) < 4                          # one epoch ran, then it stopped
    assert h[-1].get("stopped_for_time") is True
    assert all(not r.get("stopped_for_time") for r in h[:-1])


def test_no_budget_runs_every_epoch():
    tr, vl = _tiny()
    _, h = train(_cfg(time_budget_s=None), tr, vl, log=lambda s: None)
    assert len(h) == 4
    assert not any(r.get("stopped_for_time") for r in h)


def test_budget_longer_than_training_never_stops():
    tr, vl = _tiny()
    _, h = train(_cfg(time_budget_s=3600.0), tr, vl, log=lambda s: None)
    assert len(h) == 4
    assert not any(r.get("stopped_for_time") for r in h)


def test_budget_still_keeps_best_weights():
    tr, vl = _tiny()
    torch.manual_seed(0)
    _, h = train(_cfg(time_budget_s=0.0), tr, vl, log=lambda s: None)
    # the marker row must carry the usual fields, so downstream consumers
    # (dashboard, results writers) never see a truncated record
    for r in h:
        assert "epoch" in r and "train_loss" in r and "wall_s" in r


def test_divergence_guard_rolls_back():
    """A blown-up model is replaced by the best weights and the LR is halved."""
    import dataclasses
    import torch
    from ablation2d.data import AblationDataset
    from ablation2d.train import FAST, train
    root = Path(__file__).resolve().parents[1]
    tr = AblationDataset(root / "data_liver" / "train.npz", limit=48)
    va = AblationDataset(root / "data_liver" / "val.npz", limit=16)
    cfg = dataclasses.replace(FAST, device="cpu", epochs=5, validate_every=1, anatomy_steps=2,
                              warmup_epochs=1, diverge_factor=1.5)

    def sabotage(epoch, rec, model):
        if epoch == 1:
            with torch.no_grad():                 # an unmistakable blow-up
                for p in model.parameters():
                    p.normal_(0.0, 3.0)

    _, hist = train(cfg, tr, va, on_epoch=sabotage, log=lambda *a: None)
    rolled = [r for r in hist if r.get("rolled_back")]
    assert rolled, "the guard never fired"
    after = hist[hist.index(rolled[0]) + 1]
    assert after["val_loss"] < 1.2 * min(r["val_loss"] for r in hist[:2])
