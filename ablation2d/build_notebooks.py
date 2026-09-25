"""Generate both notebook variants from one source.

    python build_notebooks.py

Writes:
    notebooks/ablation_with_nca_TUTORIAL.ipynb   blanks for participants
    notebooks/ablation_with_nca_SOLUTION.ipynb   complete, runs top to bottom

A code cell may carry a `solution` body and a `stub` body; everything else is
shared, so the two files cannot drift apart. Edit this file, not the notebooks.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"

REPO = "https://github.com/JonasMht/miccai2026-nca-tutorial"

# The reference model's numbers are read from its scored checkpoint, so the text
# cannot drift from the file (re-score with scripts/score_checkpoint.py).
_scored = json.loads((ROOT / "checkpoints" / "ablation_cnca_liver.scored.json").read_text())
REF = {
    "epochs": _scored["config"]["epochs"],
    "dsc1": _scored["test"]["C-NCA"]["dice"],
    "f1_1": _scored["test"]["C-NCA"]["vessel_f1"],
    "sphere": _scored["test"]["device-chart sphere"]["dice"],
    "K": _scored["infer_steps"],
    "blocks": _scored["config"]["n_sub_models"],
}
import torch as _torch
REF["params"] = sum(v.numel() for v in _torch.load(ROOT / "checkpoints" / "ablation_cnca_liver.pt",
                                                     map_location="cpu", weights_only=False)["state_dict"].values())


def md(text):
    return {"kind": "md", "src": text}


def code(solution, stub=None):
    return {"kind": "code", "solution": solution, "stub": stub}


CELLS = [
    md("""# Ablation planning with a neural cellular automaton
**MICCAI 2026 · Learning to Self-Organize · Hands-on 3** · Jonas Mehtali, University of Strasbourg / ICube

A microwave needle in the liver, a power, five minutes of heating. Which tissue
dies? The physics answer couples Pennes bioheat, the microwave power deposition
and an Arrhenius damage integral. The solvers from my PhD work (supervised by
Prof. Caroline Essert and Juan Verde) take about **20 seconds** per plan, and
they produced every label in this notebook.

In the next 30 minutes you will train an NCA with about **ten thousand
parameters** that answers in milliseconds. From the same rollout it also
segments the liver vessels, without ever being told where they are. Vessels
carry heat away, so a model that gets the necrosis right near a vessel has to
find the vessel first.

| | | |
|---|---|---|
| 0 | setup | 2 min |
| 1 | play with the finished model | 3 min |
| 2 | one case, inputs and targets | 2 min |
| 3 | build the automaton (TODO 1-5) | 8 min |
| 4 | train it | 5 min |
| 5 | how good is it? (TODO 6) | 5 min |
| 6 | your model in the planner, and breaking it | 5 min |

**Before anything else: Runtime → Change runtime type → T4 GPU.**"""),

    md("## 0 · Setup"),
    code(f"""import os, sys, shutil, subprocess, pathlib, dataclasses

REPO = "{REPO}"
if "google.colab" in sys.modules and not pathlib.Path("miccai2026-nca-tutorial/ablation2d").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", REPO], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone failed:\\n" + r.stderr.strip())

ROOT = next((p.resolve() for p in map(pathlib.Path, ("miccai2026-nca-tutorial/ablation2d", "ablation2d", ".", ".."))
             if (p / "ablation2d" / "channels.py").exists()), None)
if ROOT is None:
    raise RuntimeError("could not find the ablation2d package next to this notebook")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU (training will be slow)")

# the test harness (scripts/test_notebook.py) sets this to run on a CPU in minutes
QUICK = os.environ.get("NCA_NOTEBOOK_QUICK") == "1"
LIMIT = 48 if QUICK else None

# ablation2d holds the plumbing: data, plotting, the training loop.
# The parts that are the lesson are typed out in this notebook.
from ablation2d import viz
from ablation2d.data import AblationDataset
from ablation2d.anatomy import from_segmentations
from ablation2d.train import FAST, load_checkpoint
from ablation2d.webplanner import BrowserPlanner, slice_from_anatomy, slices_from_split

# plans with one or two needles, as in the planner
data = {{s: AblationDataset(ROOT / "data_liver" / f"{{s}}.npz", device=DEVICE, limit=LIMIT, max_needles=2)
        for s in ("train", "val", "test")}}
scanned = AblationDataset(ROOT / "data" / "test.npz", limit=LIMIT, max_needles=2)   # same cases, CT not masked
for s, d in data.items():
    print(f"{{s:5s}} {{len(d):4d}} plans")"""),

    md(f"""## 1 · The finished model

This is the reference model, trained for {REF['epochs']} epochs on the same data you will
use: {REF['params']:,} parameters, {REF['blocks']} blocks per step, {REF['K']} steps once the anatomy is read.
Test DSC {REF['dsc1']:.3f} on the necrosis and F1 {REF['f1_1']:.3f} on
the vessels. It runs **in this page, on your GPU**: the weights were copied
into a WebGL shader, so every edit reruns the whole rollout in a few
milliseconds without going back to Python. The rollout stops at the number of
steps chosen on validation, as the model is evaluated.

Things to try:

* drag the **white dot** to move the needle, the grey one to turn it;
* park the tip right next to a large vessel. The lesion is notched on that side:
  blood carries the heat away. Nobody told the model where the vessels are;
* **+ add a needle** (or double-click) for a second one, and set each power;
* press **replay growth** to watch the field grow from the needle, one local
  update at a time;
* turn on the **eraser** and wipe part of the lesion: the automaton repairs it;
* switch **patient**: these are test slices the model never saw."""),

    code("""reference, ref_cfg, ref_ck = load_checkpoint(ROOT / "checkpoints" / "ablation_cnca_liver.pt", device=DEVICE)

real = np.load(ROOT / "data" / "real_slice.npz")
anat = from_segmentations(real["seg_liver"], real["seg_vessel"], hounsfield=real["hounsfield"])
slices = [slice_from_anatomy(anat, "demo slice")] + slices_from_split(scanned, data["test"], n=4)

BrowserPlanner([(f"reference, {ref_cfg.epochs} epochs", reference, ref_ck["vessel_threshold"], ref_cfg.infer_steps)],
               slices)"""),

    md("""## 2 · One case

Each training case is a CT slice from a real patient, one to four needles and a
power per needle. The corpus has 4 000 plans on 2 000 slices from 50 livers,
split **by patient** so that no slice of a test liver was seen in training. We
use the plans with one or two needles, 83 % of them, which is also what the
planner allows.

The model gets three channels and must produce two:

| in | | out | |
|---|---|---|---|
| `hounsfield` | the CT, zeroed outside the liver | `cell_death` | Arrhenius damage |
| `needle` | the applicator shaft | `vessel` | where the vessels are |
| `applicator_activation` | power at the radiating slot | | |

**Why mask the CT to the liver?** Three quarters of the slice is bowel, spine,
ribs and air, and for a ten-thousand-parameter model all of it is a
distraction. Masking was one of the largest effects I measured: necrosis DSC
0.848 → 0.868 and vessel F1 0.611 → 0.749. It helps a plain Hounsfield threshold
even more (vessel F1 0.245 → 0.439), so baselines are always scored on the
same masked input.

Burn time is fixed at 5 minutes, so it is not an input. The label is solved in
3-D and then sliced: a one-voxel-thick 2-D solve is an infinite slab and gives
lesions 14 % too long."""),

    code("""i = 3
x, y = data["test"].sample(i)
viz.show_case(x[0].cpu(), y[0].cpu(), hu_scanned=scanned.sample(i)[0][0, 0].numpy() * 2000 - 1000)
plt.show()
print(f"necrosis {(y[0, 0] >= 0.99).sum().item() * 0.04:.1f} cm²  ·  vessels on "
      f"{100 * y[0, 1].mean().item():.1f} % of the slice")"""),

    md("""## 3 · The automaton

A **chained** NCA: one step runs $N$ small convolution blocks in sequence, each
adding a residual update. The state is one tensor whose channels have jobs:

| channels | role | updated by the model? |
|---|---|---|
| 0 - 1 | the answers: necrosis, vessels | yes |
| 2 - 4 | the environment: CT, needle, power | no, rewritten from the inputs after every step |
| 5 + | scratch space | yes |

Each block is two 3×3 convolutions, so information travels $2N$ cells per step.
With 2 answers and 3 environment channels pinned, the scratch channels are the
only place the model can compute: a 6-channel version (one scratch channel)
scored necrosis DSC 0.30, and adding channels lifted **both** heads together.

**Anatomy first, then heat.** Where the vessels are does not depend on the
needle. A single rollout that sees the plan while it looks for them learns a
shortcut anyway: inside a lesion, it stops seeing vessels (on a single-phase
model, 8 % of the vessel pixels inside the lesion were found, against 84 %
away from it). So the rollout has two phases with the same rule: a few steps
with the needle and power channels empty, which can only find the vessels from
the CT, then the planned steps with the vessel answer held fixed, like the CT.
The vessel map cannot depend on the plan, by construction.

Every cell updates at every step. The original Growing-NCA recipe lets each
cell fire with probability ½ instead; on this task that measured worse (5-minute
budget, three seeds: necrosis DSC 0.818 → 0.831 and vessel F1 0.674 → 0.715
without it), and without it the model is deterministic.

Fill in TODO 1 to 3 below. Everything else in the class is given, including
the two phases in `forward`."""),

    code("""N_OUT, N_INPUT = 2, 3
OUT = slice(0, N_OUT)                  # necrosis, vessels
ENV = slice(N_OUT, N_OUT + N_INPUT)    # CT, needle, power
STATE_CLAMP = 4.0"""),

    code(
        solution="""class AblationCNCA(nn.Module):
    def __init__(self, channels=12, hidden_mult=2, n_sub_models=2, anatomy_steps=6):
        super().__init__()
        self.channels, self.n_sub_models = channels, n_sub_models
        self.anatomy_steps = anatomy_steps
        hidden = channels * hidden_mult
        self.sub_models = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, 3, padding=1, bias=False),
            ) for _ in range(n_sub_models)
        ])
        for sub in self.sub_models:
            fan_in = sub[0].weight[0].numel()
            nn.init.normal_(sub[0].weight, 0.0, (2.0 / fan_in) ** 0.5 * 0.1)
            nn.init.zeros_(sub[0].bias)
            nn.init.zeros_(sub[2].weight)          # every block starts as a no-op

    def step(self, x):
        for sub in self.sub_models:
            x = STATE_CLAMP * torch.tanh((x + sub(x)) / STATE_CLAMP)
        return x

    def seed(self, inputs):
        b, _, h, w = inputs.shape
        x = inputs.new_zeros(b, self.channels, h, w)
        x[:, ENV] = inputs
        return x

    def run(self, x, env, steps, hold=None, trace=None):
        # `steps` updates; after each one the environment is put back, and
        # `hold` (if given) is written back into the vessel channel
        for _ in range(steps):
            x = self.step(x)
            x = torch.cat([x[:, OUT], env, x[:, ENV.stop:]], dim=1)
            if hold is not None:
                x = torch.cat([x[:, :1], hold, x[:, 2:]], dim=1)
            if trace is not None:
                trace.append(torch.sigmoid(x[:, OUT]).detach())
        return x

    def forward(self, inputs, steps=8, state=None, return_trace=False, return_state=False):
        if state is None:
            if self.anatomy_steps:
                # phase 1, the anatomy: the needle is not in yet, so the vessels
                # are found from the CT alone
                ct_only = inputs.clone()
                ct_only[:, 1:] = 0
                state = self.run(self.seed(ct_only), ct_only, self.anatomy_steps)
            else:
                state = self.seed(inputs)
        # phase 2, the heating: the needle is on, the vessels found above are held
        hold = state[:, 1:2] if self.anatomy_steps else None
        trace = [] if return_trace else None
        x = self.run(state, inputs, steps, hold=hold, trace=trace)
        out = torch.sigmoid(x[:, OUT])
        if return_trace:
            return (out, trace, x) if return_state else (out, trace)
        return (out, x) if return_state else out


model = AblationCNCA(FAST.channels, FAST.hidden_mult, FAST.n_sub_models).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters · {model.channels} channels = {N_OUT} answers + "
      f"{N_INPUT} environment + {model.channels - N_OUT - N_INPUT} scratch")
print(f"information travels {2 * model.n_sub_models} cells (= {4 * model.n_sub_models} mm) per step")""",
        stub="""class AblationCNCA(nn.Module):
    def __init__(self, channels=12, hidden_mult=2, n_sub_models=2, anatomy_steps=6):
        super().__init__()
        self.channels, self.n_sub_models = channels, n_sub_models
        self.anatomy_steps = anatomy_steps
        hidden = channels * hidden_mult
        self.sub_models = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, 3, padding=1, bias=False),
            ) for _ in range(n_sub_models)
        ])
        for sub in self.sub_models:
            fan_in = sub[0].weight[0].numel()
            nn.init.normal_(sub[0].weight, 0.0, (2.0 / fan_in) ** 0.5 * 0.1)
            nn.init.zeros_(sub[0].bias)
            nn.init.zeros_(sub[2].weight)          # every block starts as a no-op

    def step(self, x):
        # TODO 1 - one CA step. For each block in self.sub_models, add its
        # residual update and bound the state softly:
        #   x = STATE_CLAMP * torch.tanh((x + sub(x)) / STATE_CLAMP)
        # The tanh is a soft clamp: a hard clamp has zero gradient once the
        # state sits on the bound, and training stops there.
        raise NotImplementedError

    def seed(self, inputs):
        # TODO 2 - the initial state: zeros of shape (B, self.channels, H, W),
        # with the 3 input channels written into x[:, ENV]. Both answers start
        # at zero: nothing is dead yet and no vessel has been found.
        raise NotImplementedError

    def run(self, x, env, steps, hold=None, trace=None):
        # `steps` updates; after each one the environment is put back, and
        # `hold` (if given) is written back into the vessel channel
        for _ in range(steps):
            x = self.step(x)
            # TODO 3 - put the environment back: the CT and the plan are not the
            # model's to edit. Rebuild x from its answers x[:, OUT], `env` and its
            # scratch channels x[:, ENV.stop:] (torch.cat along dim=1).
            raise NotImplementedError
            if hold is not None:
                x = torch.cat([x[:, :1], hold, x[:, 2:]], dim=1)
            if trace is not None:
                trace.append(torch.sigmoid(x[:, OUT]).detach())
        return x

    def forward(self, inputs, steps=8, state=None, return_trace=False, return_state=False):
        if state is None:
            if self.anatomy_steps:
                # phase 1, the anatomy: the needle is not in yet, so the vessels
                # are found from the CT alone
                ct_only = inputs.clone()
                ct_only[:, 1:] = 0
                state = self.run(self.seed(ct_only), ct_only, self.anatomy_steps)
            else:
                state = self.seed(inputs)
        # phase 2, the heating: the needle is on, the vessels found above are held
        hold = state[:, 1:2] if self.anatomy_steps else None
        trace = [] if return_trace else None
        x = self.run(state, inputs, steps, hold=hold, trace=trace)
        out = torch.sigmoid(x[:, OUT])
        if return_trace:
            return (out, trace, x) if return_state else (out, trace)
        return (out, x) if return_state else out


model = AblationCNCA(FAST.channels, FAST.hidden_mult, FAST.n_sub_models).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters · {model.channels} channels = {N_OUT} answers + "
      f"{N_INPUT} environment + {model.channels - N_OUT - N_INPUT} scratch")
print(f"information travels {2 * model.n_sub_models} cells (= {4 * model.n_sub_models} mm) per step")"""),

    md("""**A free check before training.** Every block's last convolution starts at
zero, so the untrained model must output an *exactly* flat field. If it does
not, the state layout or the step is wrong, and you find out in two seconds
instead of after five minutes of training."""),

    code(
        solution="""with torch.no_grad():
    p = model(data["val"].sample(0)[0], steps=20)
spread = float(p.max() - p.min())
print(f"output {tuple(p.shape)}, spread {spread:.1e}:", "flat, good" if spread == 0 else "NOT flat")""",
        stub="""# TODO 4 - run the untrained model for 20 steps on one validation case
# (data["val"].sample(0)[0]) under torch.no_grad(). Print the output shape and
# float(p.max() - p.min()). Expect (1, 2, 128, 128) and exactly 0.0.
raise NotImplementedError"""),

    md("""### The loss

Necrosis covers 1 to 4 % of the slice, and an unweighted MSE is minimised by
predicting nothing. So foreground cells weigh **100 : 1**: a missed necrotic
cell costs as much as a hundred invented ones. For an ablation margin,
over-predicting is the safer mistake, and that ratio is a clinical choice.

The vessels get Dice + weighted BCE, and the two are summed as
`L = L_necrosis + w · L_vessel`. Remember that `w`, we come back to it."""),

    code(
        solution="""def necrosis_loss(pred, target, fg=100.0, bg=1.0, thr=0.01):
    w = torch.where(target > thr, torch.full_like(target, fg), torch.full_like(target, bg))
    return (w * (pred - target) ** 2).sum() / w.sum()""",
        stub="""def necrosis_loss(pred, target, fg=100.0, bg=1.0, thr=0.01):
    # TODO 5 - weighted MSE: the weight is `fg` where target > thr and `bg`
    # elsewhere (torch.where + torch.full_like). Return the weighted mean,
    # (w * (pred - target) ** 2).sum() / w.sum()
    raise NotImplementedError"""),

    code("""# two cases you can check in your head
p_, t_ = torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[[1.0, 1.0]]]])
assert abs(float(necrosis_loss(p_, t_)) - 0.5) < 1e-6      # weights 100, 100; errors 1, 0
t_ = torch.tensor([[[[1.0, 0.0]]]])
assert abs(float(necrosis_loss(p_, t_)) - 1.0) < 1e-6      # weights 100, 1; errors 1, 1
print("necrosis_loss is correct")"""),

    md("""## 4 · Training

The loop is imported, it is plumbing. The tricks from the talk are inside it,
each marked `# TRICK n` in `ablation2d/train.py`: a random rollout length per
batch, starting some batches from states the model reaches on its own,
per-parameter gradient normalisation, and persistence (a second, longer stretch
of the same rollout is supervised too, so the field holds when you keep
iterating, which the planner relies on).

The budget is **5 minutes of wall clock**, not a number of epochs, so it works
on any GPU. The learning rate follows the clock and the best validation epoch
is kept.

Four settings below were measured, not guessed:

* `lr = 5e-3`, higher than the long recipe uses. With only about fifteen epochs
  to spend, it gave the vessel head +0.07 F1 over three seeds, for no
  measurable cost on the necrosis.

* `vessel_weight = 0.3`. At 3.0 the necrosis term received only 3 % of the
  loss, and moving this one number took validation DSC at epoch 10 from 0.46
  to 0.79. You will see the shares on your own model in section 5.
* `ema_decay = 0.99`: the model that is kept is a running average of the
  weights, which is less noisy than the last step. Together with the one-or-two-needle
  plans, it took the 5-minute model from DSC 0.83 to 0.86.
* `augment_d4 = False`. Rotations and flips look free for a diffusion problem,
  but these are axial CT slices in a standard orientation. With augmentation the
  vessel head fell from F1 0.70 to 0.28 while the necrosis did not move.

No GPU? Set `LOAD_PRETRAINED = True` to load the reference model instead."""),

    code("""from ablation2d.train import train
from ablation2d.live import LiveDashboard

LOAD_PRETRAINED = os.environ.get("NCA_NOTEBOOK_PRETRAINED") == "1"   # set True to skip training

cfg = dataclasses.replace(FAST, device=DEVICE, lr=5e-3, vessel_weight=0.3, augment_d4=False,
                          anatomy_steps=model.anatomy_steps, ema_decay=0.99,
                          epochs=24, warmup_epochs=2, validate_every=2, time_budget_s=5 * 60)
if QUICK:
    cfg = dataclasses.replace(cfg, epochs=2, validate_every=1, time_budget_s=None)

if LOAD_PRETRAINED:
    model, cfg, _ = load_checkpoint(ROOT / "checkpoints" / "ablation_cnca_liver.pt", device=DEVICE)
else:
    dash = LiveDashboard(data["val"], steps=cfg.infer_steps, budget_s=cfg.time_budget_s)
    model, history = train(cfg, data["train"], data["val"], model=model, on_epoch=dash,
                           select="both", log=lambda *a: None)"""),

    md("""**While it trains**, three things to predict. The answers are in the
curves above and in section 5.

1. Which head converges first, necrosis or vessels? Why would one be harder?
2. Which of the three inputs matters most? Guess now, then check in the
   planner in section 6 by setting a needle to 0 W.
3. It is trained with rollouts of 3 to 10 steps. What happens at 400?"""),

    md("""## 5 · How good is it?

The number of steps at inference is a free choice that training does not fix.
Pick it on validation, and take the cheapest value within 1 % of the best."""),

    code(
        solution="""from ablation2d.train import choose_inference_steps

K, scores = choose_inference_steps(model, data["val"], candidates=range(2, 25, 2))
cfg = dataclasses.replace(cfg, infer_steps=K)

plt.figure(figsize=(6, 2.6), dpi=100)
plt.plot(list(scores), list(scores.values()), "-o", ms=4, color="#d9480f")
plt.axvline(K, color="#888", ls="--", lw=1)
plt.xlabel("rollout steps K"); plt.ylabel("val DSC"); plt.title(f"chosen K = {K}")
plt.grid(alpha=0.3); plt.show()""",
        stub="""from ablation2d.train import choose_inference_steps

# TODO 6 - sweep the rollout length on VALIDATION and keep the cheapest K
# within 1 % of the best:
#     K, scores = choose_inference_steps(model, data["val"], candidates=range(2, 25, 2))
#     cfg = dataclasses.replace(cfg, infer_steps=K)
# then plot scores (a dict K -> DSC) if you like.
raise NotImplementedError"""),

    md("""Now the test set, against two baselines that do not need a network:

* **the device chart**: a sphere sized from power and time, which is roughly
  what the manufacturer's table gives a clinician. It ignores the anatomy, so
  beating it means the model learned something about the anatomy;
* **the best Hounsfield threshold** for the vessels, fitted on the split it is
  scored on. Contrast-enhanced vessels are bright, so `hu > t` is a working
  segmenter.

The vessel threshold is chosen on validation, like K."""),

    code("""import pandas as pd
from ablation2d.evaluate import fit_sphere_baseline, predict_all, report, sphere_baseline, hu_threshold_vessel_f1
from ablation2d.model import best_vessel_threshold

sphere = fit_sphere_baseline(data["train"])
with torch.no_grad():
    vthr, _ = best_vessel_threshold(predict_all(model, data["val"], K), data["val"].targets)

def row(r):
    return {"necrosis DSC": r.get("dice"), "area error cm²": r.get("area_err_cm2"),
            "recall": r.get("recall"), "precision": r.get("precision"), "vessel F1": r.get("vessel_f1")}

one = report(model, data["test"], steps=K, vessel_threshold=vthr,
             baselines={"device chart": lambda x: sphere_baseline(x, sphere)})
table = pd.DataFrame({"your model": row(one["C-NCA"]),
                      "device-chart sphere": row(one["device chart"]),
                      "best HU threshold": {"vessel F1": hu_threshold_vessel_f1(data["test"])}}).T
table.style.format(precision=3, na_rep="")"""),

    md(f"""**Read recall against precision.** Recall above precision means the model
over-predicts the zone, which is what the 100 : 1 weighting asked for. The
reference model scores DSC {REF['dsc1']:.3f} and vessel F1 {REF['f1_1']:.3f} on the same split.

### Check the share, not the weight

The two loss terms live on different scales. Here is how much of the total
the necrosis head actually receives on your model, at three weights."""),

    code("""from ablation2d.model import soft_dice_loss, masked_bce

xb, yb = data["val"].sample(slice(0, 16))
with torch.no_grad():
    p = model(xb, steps=K)
l_nec = float(necrosis_loss(p[:, :1], yb[:, :1]))
l_ves = float(soft_dice_loss(p[:, 1:], yb[:, 1:]) + masked_bce(p[:, 1:], yb[:, 1:], 20.0))
for w in (3.0, 1.0, 0.3):
    print(f"w = {w:3.1f}   necrosis gets {100 * l_nec / (l_nec + w * l_ves):4.1f} % of the loss")"""),

    md("""### The second head

Best, median and worst test cases for the vessel head. Green is vessel found,
blue missed, red invented. Expect the large vessels to be found and the fine
branches to be missed: that is where most of the remaining error is."""),

    code("""from ablation2d.model import vessel_f1

with torch.no_grad():
    pt = predict_all(model, data["test"], K)
per = vessel_f1(pt, data["test"].targets, vthr)
order = torch.argsort(per, descending=True)
pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
viz.vessel_gallery([data["test"].sample(k)[0][0, 0].cpu() for k in pick],
                   [data["test"].targets[k, 1].cpu() for k in pick],
                   [pt[k, 1].cpu() for k in pick], vthr, f1=[float(per[k]) for k in pick])
plt.show()"""),

    md(f"""## 6 · Your model in the planner

Both models are in the **model** menu now: yours after five minutes, and the
reference after {REF['epochs']} epochs. Same slices, same needles.

Try to break them. Predict what happens before you try each one:

* **Too much power.** The corpus was sampled between 30 and 150 W. Scroll a
  needle up to 250 W.
* **Outside the liver.** Drag the tip into the stomach or beside the spine. The
  model was never shown a tip there, and with the masked CT it cannot see the
  tissue it would be burning.
* **Carry the state across edits.** Tick it, then move the needle. The rollout
  now continues from the old answer instead of the blank seed. The model was
  trained to *hold* an answer once it has one: how far does the lesion follow?
* **Look inside.** In the **show** menu pick a scratch channel (5 and up).
  These have no units and nobody told them what to do.

No live view (an old browser without WebGL2)? `AblationStudio` from
`ablation2d.ui` is the same planner with sliders, rendered in Python."""),

    code("""BrowserPlanner([("your model, 5 minutes", model, vthr, K),
                (f"reference, {ref_cfg.epochs} epochs", reference, ref_ck["vessel_threshold"], ref_cfg.infer_steps)],
               slices)"""),

    md("""One number to go with the "keep state" experiment: the same rollout run far
past the lengths it was trained on."""),

    code("""from ablation2d.model import dice

xb, yb = data["test"].sample(slice(0, 32))
with torch.no_grad():
    for k in (K, 30, 100, 400):
        print(f"K = {k:3d}   DSC {float(dice(model(xb, steps=k), yb).mean()):.3f}")"""),

    md(f"""## What to take away

* **Heat diffusion is a local rule iterated in time, and so is an NCA.** That
  match is why ten thousand parameters can stand in for a 20-second solve.
* **One rule, two outputs.** The same rollout predicts the necrosis and
  segments the vessels it needs, from a CT, a needle and a power.
* **Freeze what must not depend on the plan.** Found from the CT alone and
  then held fixed, the vessel map cannot be bent by the lesion; seen together
  with the plan, the model had learned to stop seeing vessels inside a lesion.
* **Scratch channels are where it computes.** Answers and environment are
  pinned; width went into both heads at once.
* **Check the share, not the weight.** A loss weight of 3.0 looked harmless and
  gave one head 3 % of the loss.
* **Show the model what matters, and measure the tricks.** Masking the CT
  helped both heads; augmentation, which looked free, cost the vessel head
  most of its F1.
* **A surrogate is only as good as its labels.** The device calibration behind
  every label here is provisional: fitted on ex vivo bovine liver at 17 °C,
  held-out error 9.9 %, not clinically validated.

**Going further:** predict peak temperature and read the 50 °C isotherm;
condition on a tumour mask and report the margin; put the surrogate inside an
optimiser and search for the plan, which is what the 3-D version does.

**Reading:** Mehtali, Verde & Essert, *C-NCA* (MICCAI 2025) — the paper behind
this hands-on · Mehtali, Verde & Essert, *Heat* (IJCARS 2025) — the 3-D solver
that made the labels · Mordvintsev et al., *Growing Neural Cellular Automata*
(Distill 2020) · Kalkhof, González & Mukhopadhyay, *Med-NCA* (IPMI 2023, TU
Darmstadt) · Pennes (1948) · Lu et al., *AJR* (2002) on the vessel heat sink.

Code, both notebooks and the slides: {REPO}"""),
]

N_TODO_CELLS = sum(1 for c in CELLS if c["kind"] == "code" and c["stub"])
N_TODOS = sum(c["stub"].count("# TODO ") for c in CELLS
              if c["kind"] == "code" and c["stub"])


def build(variant: str):
    cells = []
    for c in CELLS:
        if c["kind"] == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "id": f"md{len(cells)}",
                          "source": c["src"].splitlines(keepends=True)})
        else:
            body = c["stub"] if (variant == "TUTORIAL" and c["stub"]) else c["solution"]
            cells.append({"cell_type": "code", "metadata": {}, "id": f"c{len(cells)}",
                          "execution_count": None, "outputs": [],
                          "source": body.splitlines(keepends=True)})

    if variant == "TUTORIAL":
        cells.insert(1, {
            "cell_type": "markdown", "metadata": {}, "id": "howto",
            "source": [
                f"> **How to use this notebook.** There are {N_TODOS} `TODO`s in {N_TODO_CELLS} cells, "
                "each a few lines with the answer spelled out in the hint.\n",
                "> Stuck? `ablation_with_nca_SOLUTION.ipynb` in the same folder is the completed version.\n",
            ],
        })

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / f"ablation_with_nca_{variant}.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")


if __name__ == "__main__":
    build("TUTORIAL")
    build("SOLUTION")
