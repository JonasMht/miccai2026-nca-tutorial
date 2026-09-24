"""The training dashboard: both heads, redrawn in place after every epoch.

Draws into one ipywidgets `Output` with `clear_output(wait=True)`, which behaves
the same in Colab, JupyterLab and VS Code.
"""
from __future__ import annotations

import time

import numpy as np


class LiveDashboard:
    """Pass as `on_epoch=` to `train.train`.

    Shows the training loss, validation DSC and vessel F1, and what the model
    currently predicts on one validation case: necrosis on the left, vessels on
    the right, with the truth as contours.
    """

    def __init__(self, val_ds, sample_index: int | None = None, steps: int = 8,
                 budget_s: float | None = None):
        from IPython.display import display
        import ipywidgets as W

        self.val = val_ds
        self.steps = steps
        self.budget_s = budget_s
        self.i = self._pick_case() if sample_index is None else sample_index
        self.hist = {"epoch": [], "loss": [], "v_epoch": [], "dice": [], "f1": []}
        self.t0 = time.time()
        self.out = W.Output()
        display(self.out)

    def _pick_case(self) -> int:
        """A case with a mid-sized lesion and plenty of vessel, so both heads show."""
        y = self.val.targets
        les = y[:, 0].float().mean((1, 2)).cpu().numpy()
        ves = y[:, 1].float().sum((1, 2)).cpu().numpy()
        ok = np.where((les > 0.015) & (les < 0.09))[0]
        return int(ok[np.argmax(ves[ok])]) if len(ok) else 0

    def __call__(self, epoch, rec, model):
        import torch
        from IPython.display import clear_output
        from matplotlib import pyplot as plt

        from . import viz

        self.hist["epoch"].append(epoch)
        self.hist["loss"].append(rec["train_loss"])
        if "val_dice" in rec:
            self.hist["v_epoch"].append(epoch)
            self.hist["dice"].append(rec["val_dice"])
            self.hist["f1"].append(rec["val_vessel_f1"])

        x, y = self.val.sample(self.i)
        was = model.training
        model.eval()
        with torch.no_grad():
            p = model(x, steps=self.steps)[0].float().cpu().numpy()
        model.train(was)
        y = y[0].cpu().numpy()
        hu = x[0, 0].cpu().numpy() * 2000.0 - 1000.0

        with self.out:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), dpi=100,
                                   gridspec_kw={"width_ratios": [1.15, 1.15, 1, 1]})
            ax[0].plot(self.hist["epoch"], self.hist["loss"], lw=1.5, color="#444")
            ax[0].set_yscale("log")
            ax[0].set_title("training loss", fontsize=10)
            ax[0].set_xlabel("epoch")

            if self.hist["v_epoch"]:
                ax[1].plot(self.hist["v_epoch"], self.hist["dice"], "-o", ms=3, lw=1.5,
                           color="#d9480f", label="necrosis DSC")
                ax[1].plot(self.hist["v_epoch"], self.hist["f1"], "-o", ms=3, lw=1.5,
                           color="#1c7ed6", label="vessel F1")
                ax[1].legend(loc="lower right", fontsize=8, frameon=False)
            ax[1].set_ylim(0, 1)
            ax[1].set_title("validation", fontsize=10)
            ax[1].set_xlabel("epoch")
            for a in ax[:2]:
                a.grid(alpha=0.25)
                a.spines[["top", "right"]].set_visible(False)

            viz.show_ct(hu, ax[2])
            ax[2].imshow(np.ma.masked_less(p[0], 0.1), cmap=viz.NECROSIS, vmin=0, vmax=1,
                         origin="upper", interpolation="nearest")
            if (y[0] >= 0.99).any():
                ax[2].contour(y[0], levels=[0.99], colors=["#00e0ff"], linewidths=1.0)
            ax[2].set_title("necrosis (cyan: truth)", fontsize=10)

            viz.show_ct(hu, ax[3])
            ax[3].imshow(np.ma.masked_less(p[1], 0.5), cmap=viz.VESSELS, vmin=0, vmax=1,
                         origin="upper", interpolation="nearest")
            ax[3].contour(y[1], levels=[0.5], colors=["#ffb300"], linewidths=0.8)
            ax[3].set_title("vessels (orange: truth)", fontsize=10)
            for a in ax[2:]:
                a.set_xticks([]); a.set_yticks([])

            t = time.time() - self.t0
            clock = f"{int(t // 60)}:{int(t % 60):02d}"
            if self.budget_s:
                b = self.budget_s
                clock += f" of {int(b // 60)}:{int(b % 60):02d}"
            head = f"epoch {epoch + 1}   ·   {clock}"
            if self.hist["dice"]:
                head += (f"   ·   best so far: DSC {max(self.hist['dice']):.3f}, "
                         f"vessel F1 {max(self.hist['f1']):.3f}")
            fig.suptitle(head, fontsize=11, x=0.01, ha="left")
            fig.tight_layout()
            plt.show()
