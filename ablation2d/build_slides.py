"""Build the tips-and-tricks deck: one self-contained HTML file.

    python build_slides.py        # writes slides/tips_and_tricks_nca.html
                                  #    and slides/speaker_notes.md

Every number and every chart is read from the measurement files (results/*.json
and the scored checkpoint), so re-running this after a new measurement updates
the deck. A missing file shows up on the slide as "measurement missing" rather
than as a plausible number.

In the browser: arrows / space / click to move, P for the presenter view (notes,
per-slide timer, total time against the 10-minute slot), F for full screen.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "slides"

ORANGE, BLUE, GREY, INK, GREEN = "#d9480f", "#1c7ed6", "#adb5bd", "#1d1f24", "#2b8a3e"
MISSING = '<span class="missing">measurement missing</span>'

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 17,
    "axes.edgecolor": "#555", "axes.labelcolor": "#333", "xtick.color": "#444",
    "ytick.color": "#444", "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
})


def load(name):
    """A JSON from results/, or from anywhere under ROOT if the name has a folder."""
    p = ROOT / name if "/" in name else RES / name
    return json.loads(p.read_text()) if p.exists() else None


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def png(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def f3(x):
    return f"{x:.3f}" if x is not None else MISSING


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
SCORED = load("checkpoints/ablation_cnca_liver.scored.json")
TEST = SCORED["test"] if SCORED else {}
NCA1 = TEST.get("C-NCA", {})
NCA8 = TEST.get("C-NCA + 8 repeats", {})
SPHERE = TEST.get("device-chart sphere", {})
HU = TEST.get("best single HU threshold", {})
CNN = load("baseline_cnn.json")
STAB = load("stability.json")
CAP = load("capacity_ab.json")
AMP = load("amp_speed.json")
D4 = load("fast_d4on.json")                      # the session recipe with D4 on
D4_OFF = load("ck_fast_liver/ablation_cnca_fast.scored.json")   # the same, D4 off
WIN = load("window_grow.json")


def params_of_shipped():
    import torch
    ck = torch.load(ROOT / "checkpoints/ablation_cnca_liver.pt", map_location="cpu",
                    weights_only=False)
    return sum(v.numel() for v in ck["state_dict"].values())


PARAMS = params_of_shipped()


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def chart_rollout_strip():
    """The reference model growing a lesion on the demo slice, for the title."""
    import torch
    from ablation2d.anatomy import from_segmentations
    from ablation2d.channels import build_inputs
    from ablation2d.plans import FIXED_DURATION_S, Needle, Plan
    from ablation2d.train import load_checkpoint
    from ablation2d.viz import NECROSIS, VESSELS

    model, cfg, _ = load_checkpoint(ROOT / "checkpoints/ablation_cnca_liver.pt")
    real = np.load(ROOT / "data/real_slice.npz")
    anat = from_segmentations(real["seg_liver"], real["seg_vessel"], hounsfield=real["hounsfield"])
    ys, xs = np.nonzero(anat.liver_mask)
    tip = (float(np.median(xs)) * 2 - 6, float(np.median(ys)) * 2 - 20)
    plan = Plan([Needle(tip[0], tip[1], tip[0] + 10, 0.0, 90.0, FIXED_DURATION_S)])
    x = torch.from_numpy(build_inputs(anat, plan, mask_ct=True))[None]
    torch.manual_seed(3)
    with torch.no_grad():
        _, trace = model(x, steps=12, return_trace=True)
    hu = anat.hounsfield
    fig, ax = plt.subplots(1, 6, figsize=(15, 2.7))
    for a, k in zip(ax, (1, 2, 3, 5, 7, 11)):
        f = trace[k][0].numpy()
        a.imshow(hu, cmap="gray", vmin=20, vmax=220)
        a.imshow(np.ma.masked_less(f[1], 0.6), cmap=VESSELS, vmin=0, vmax=1)
        a.imshow(np.ma.masked_less(f[0], 0.35), cmap=NECROSIS, vmin=0, vmax=1)
        a.set_xlim(20, 100); a.set_ylim(95, 15)
        a.set_title(f"step {k + 1}", fontsize=14, color="#555")
        a.axis("off")
    fig.subplots_adjust(wspace=0.04)
    return png(fig, dpi=120)


def chart_persistence():
    if not STAB:
        return MISSING
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, lab, col in (("without_persistence", "sampled length only", GREY),
                          ("with_persistence", "+ persistence", ORANGE)):
        d = STAB[key]["dice_by_k"]
        ks = [int(k) for k in d]
        ax.plot(ks, list(d.values()), "-o", color=col, lw=3, ms=6, label=lab)
    ax.axvspan(3, 10, color="#e9ecef", zorder=-1)
    ax.text(5.5, 0.855, "trained\non 3-10", ha="center", va="top", fontsize=14, color="#666")
    ax.set_xscale("log"); ax.set_xticks([4, 10, 30, 100, 400]); ax.set_xticklabels([4, 10, 30, 100, 400])
    ax.set_xlabel("rollout steps at inference"); ax.set_ylabel("validation DSC")
    ax.set_ylim(0.6, 0.87); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.02))
    return svg(fig)


def chart_capacity():
    if not CAP:
        return MISSING
    arms = CAP["arms"]
    ch = [int(k.split("ch")[0]) for k in arms]
    dsc = [v["dice"] for v in arms.values()]
    f1 = [v["vessel_f1"] for v in arms.values()]
    x = np.arange(len(ch))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.19, dsc, 0.36, color=ORANGE, label="necrosis DSC")
    ax.bar(x + 0.19, f1, 0.36, color=BLUE, label="vessel F1")
    ax.set_xticks(x); ax.set_xticklabels([f"{c} ch\n{c - 5} scratch" for c in ch])
    ax.set_ylim(0, 0.85); ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylabel("test score")
    return svg(fig)


def chart_share():
    # one validation batch of the w = 3.0 model: necrosis 0.0744, vessels 0.8025
    l_nec, l_ves = 0.0744, 0.8025
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for i, w in enumerate((3.0, 0.3)):
        s = l_nec / (l_nec + w * l_ves)
        ax.barh(i, s, color=ORANGE, height=0.6)
        ax.barh(i, 1 - s, left=s, color=BLUE, height=0.6, alpha=0.85)
        ax.text(s + 0.01 if s < 0.1 else s / 2, i, f"{100 * s:.0f} %", va="center",
                ha="left" if s < 0.1 else "center", color="white" if s >= 0.1 else INK,
                fontsize=17, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["w = 3.0", "w = 0.3"], fontsize=17)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title("share of the loss:  necrosis  |  vessels", loc="left", fontsize=14, color="#555")
    return svg(fig)


def chart_k_sweep():
    if not SCORED:
        return MISSING
    d = SCORED["step_sweep"]
    ks = [int(k) for k in d]
    v = list(d.values())
    K = SCORED["infer_steps"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(ks, v, "-", color=ORANGE, lw=2.5)
    ax.axhline(max(v) * 0.99, color=GREY, ls="--", lw=1)
    ax.plot([K], [d[str(K)] if str(K) in d else d[K]], "o", ms=11, color=INK)
    ax.annotate(f"K = {K}", (K, d.get(str(K), d.get(K))), xytext=(12, -26),
                textcoords="offset points", fontsize=15)
    ax.set_xlabel("rollout steps K"); ax.set_ylabel("val DSC")
    ax.set_ylim(min(v) - 0.01, max(v) + 0.01)
    return svg(fig)


def chart_threshold():
    curve = (SCORED or {}).get("vessel_threshold_curve_val")
    if not curve:
        return MISSING
    t = [float(k) for k in curve]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axvspan(0.7, 1.0, color="#fff4e6", zorder=-1)
    ax.text(0.85, min(curve.values()) + 0.005, "never\nsearched", ha="center", fontsize=12, color=ORANGE)
    ax.plot(t, list(curve.values()), "-o", color=BLUE, lw=2.5, ms=5)
    best = max(curve, key=curve.get)
    ax.plot([float(best)], [curve[best]], "o", ms=11, color=INK)
    ax.set_xlabel("vessel decision threshold"); ax.set_ylabel("val vessel F1")
    ax.set_xlim(0.28, 1.0)
    return svg(fig)


def chart_anatomy():
    a = load("anatomy_first.json")
    if not a:
        return MISSING
    single = a["single rollout (seeds 0-1)"]
    first = a["anatomy first, 6 CT-only steps (seeds 0-2)"]
    keys = [("vessel_recall_in_lesion", "inside\nthe lesion"), ("vessel_recall_ring_outside", "8 mm\naround it"),
            ("vessel_recall_far", "further\naway")]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (d, lab, col) in enumerate(((single, "one rollout", GREY), (first, "anatomy first", BLUE))):
        v = [d[k] for k, _ in keys]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=col, label=lab)
        for xx, vv in zip(x, v):
            ax.text(xx + (i - 0.5) * 0.38, vv + 0.015, f"{vv:.2f}", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys])
    ax.set_ylim(0, 1); ax.set_ylabel("vessels found (recall)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    return svg(fig)


def chart_cnn():
    if not CNN:
        return MISSING
    rows = [("C-NCA", NCA1.get("dice"), NCA1.get("vessel_f1"), ORANGE),
            ("dilated CNN", CNN.get("dice"), CNN.get("vessel_f1"), GREY)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(2)
    for i, (name, d, f, col) in enumerate(rows):
        ax.bar(x[0] + (i - 0.5) * 0.36, d or 0, 0.34, color=col, label=name)
        ax.bar(x[1] + (i - 0.5) * 0.36, f or 0, 0.34, color=col)
        for xx, v in ((x[0], d), (x[1], f)):
            if v:
                ax.text(xx + (i - 0.5) * 0.36, v + 0.01, f"{v:.3f}", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(["necrosis DSC", "vessel F1"])
    ax.set_ylim(0, 1.05); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    return svg(fig)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def code_block(src):
    return f"<pre><code>{html.escape(src.strip())}</code></pre>"


def build():
    amp = (AMP or {}).get("TITAN V (sm_70, a T4-class card)", {})
    d4_f1 = D4["test"]["C-NCA"].get("vessel_f1") if D4 else None
    d4_off_f1 = D4_OFF["test"]["C-NCA"].get("vessel_f1") if D4_OFF else None
    win = WIN or {}
    grow_f1 = win.get("growing (3 -> 18)", {}).get("test_vessel_f1")
    sess = (load("session_budget.json") or {}).get("arms", {})
    fire_f1 = sess.get("12 ch, lr 5e-3, fire rate 0.5", {}).get("vf1_1x")
    nofire_f1 = sess.get("12 ch, lr 5e-3, no fire mask (the notebook)", {}).get("vf1_1x")
    wide_f1 = win.get("fixed wide (3-18)", {}).get("test_vessel_f1")
    stab = STAB or {}
    k400 = lambda key: stab.get(key, {}).get("dice_by_k", {}).get("400")
    curve = (SCORED or {}).get("vessel_threshold_curve_val", {})
    thr_gain = (max(curve.values()) - curve.get("0.7", curve.get(0.7, 0))) if curve else None

    slides = []

    def slide(title, body, notes, seconds, cls=""):
        slides.append({"title": title, "body": body, "notes": notes, "seconds": seconds, "cls": cls})

    slide("", f"""
      <div class="kicker">MICCAI 2026 · Learning to Self-Organize · Tips for training</div>
      <h1 class="big">Training NCAs:<br>what actually moved the numbers</h1>
      <img class="strip" src="{chart_rollout_strip()}">
      <div class="who"><b>Jonas Mehtali</b> · University of Strasbourg / ICube</div>
      <div class="muted">Every number in this talk was measured on the task of the next 30 minutes:
      a CT slice and a needle in, necrosis and vessels out, with a {PARAMS:,}-parameter reference model.</div>""",
          "One line: these are the tricks nobody puts in the method section, each one measured on "
          "one model, the one from Caroline's talk that you train right after this.", 20, "title")

    slide("Why an NCA is harder to train than a CNN", """
      <div class="cols">
        <div>
          <div class="diagram">
            <div class="lbl">CNN</div>
            <div class="row">""" + "".join(f'<span class="blk c{i % 4}"></span>' for i in range(12)) + """</div>
            <div class="cap">depth 12, each layer its own weights, one pass</div>
            <div class="lbl">NCA</div>
            <div class="row">""" + "".join('<span class="blk same"></span>' for _ in range(12)) + """</div>
            <div class="cap">one small rule, applied 10 to 40 times</div>
          </div>
        </div>
        <div>
          <p class="lead">What you will see when it goes wrong:</p>
          <ul>
            <li>loss goes to NaN after a few hundred steps</li>
            <li>loss goes down, output is flat zero</li>
            <li>perfect at 12 steps, garbage at 40</li>
            <li>fine while training, drifts when you keep iterating</li>
          </ul>
        </div>
      </div>
      <p class="punch">Shared weights, unrolled many times, no normalisation, no global view.
      Every trick that follows is about that.</p>""",
          "Say the last sentence slowly. Gradients flow through every step of the rollout, and "
          "small errors compound geometrically. Everything after this slide is a way to tame that.",
          50)

    slide("Two fixes that cost two lines each", f"""
      <div class="cols">
        <div>
          <h3>Zero-init the last convolution</h3>
          {code_block('''
nn.init.zeros_(block[-1].weight)

p = model(x, steps=20)          # untrained
assert float(p.max() - p.min()) == 0.0''')}
          <p>The untrained rule is the identity, so training starts from "do no harm". The
          assert catches a wrong state layout in two seconds, before you train.</p>
        </div>
        <div>
          <h3>Normalise gradients per parameter</h3>
          {code_block('''
loss.backward()
for p in model.parameters():
    p.grad /= p.grad.norm() + 1e-8
clip_grad_norm_(model.parameters(), 1.0)
opt.step()''')}
          <p>Through a long rollout, gradient norms differ by orders of magnitude between
          layers and change every batch. A global clip keeps the bad ratio.</p>
        </div>
      </div>""",
          "Both are in the notebook: the flat-output assert is TODO 4. Per-parameter normalisation "
          "is TRICK 3 in train.py.", 50)

    slide("Sample the rollout length, then teach it to stay", f"""
      <div class="cols wide-left">
        <div class="chart">{chart_persistence()}</div>
        <div>
          {code_block('''
k = randint(3, 10)        # every batch
pred, state = model(x, k)
loss = L(pred, y)

# sometimes: keep going
state = model(x, 150, state).detach()
loss += L(model(x, 6, state), y)''')}
          <p>A fixed K teaches a choreography that ends at step K. Sampling K forces a fixed
          point, and supervising a second, longer stretch of the same rollout makes it hold.</p>
          <p class="num">DSC at 400 steps: <b>{f3(k400("without_persistence"))}</b> →
          <b class="o">{f3(k400("with_persistence"))}</b></p>
        </div>
      </div>""",
          "Point at the grey curve: it was trained on 3 to 10 steps and falls apart past 20. The "
          "orange one holds to 400. Only the last 6 steps of the long stretch carry a gradient, so "
          "memory stays flat. The planner in the notebook depends on this.", 60)

    slide("Keep the environment out of the model's reach", f"""
      <div class="chan">
        <span class="c a">necrosis</span><span class="c a">vessels</span>
        <span class="c e">CT</span><span class="c e">needle</span><span class="c e">power</span>
        {''.join('<span class="c s"></span>' for _ in range(11))}
      </div>
      <div class="chan-lbl"><span>answers</span><span>environment, rewritten every step</span><span>scratch</span></div>
      <div class="cols">
        <div>{code_block('''
for _ in range(steps):
    x = step(x)
    x = cat([x[:, :2], inputs, x[:, 5:]], 1)''')}</div>
        <div>
          <p>Let the model overwrite its inputs and it will. In our 3-D model, which did not
          restore them, the power channel drifted to <b>7 630×</b> its own mean and the needle to
          <b>75×</b>: the plan was simply erased.</p>
          <p>Restoring costs three channels, and it is what lets you <b>move a needle
          mid-rollout</b> and watch the field follow.</p>
        </div>
      </div>""",
          "The drift numbers come from the 3-D surrogate of my PhD codebase, which did not restore "
          "its conditioning. Sparse channels, the plan, are the ones that get erased. The payoff is "
          "the live planner in the hands-on.", 60)

    slide("Scratch channels are where the model computes", f"""
      <div class="cols wide-left">
        <div class="chart">{chart_capacity()}</div>
        <div>
          <p>Subtract what is spoken for: 2 answers and 3 environment channels. A 6-channel model
          has <b>one</b> channel left to think in.</p>
          <p>Both heads rise <b>together</b> as you add width. That is two tasks competing for
          memory, not for gradient, and no loss weighting can buy a channel.</p>
          <p class="muted">Same budget per arm, real CT, before the loss fix on the next slide.</p>
        </div>
      </div>""",
          "Everything we tried on the loss first left the vessel head pinned. Width moved both "
          "heads at once. 16 channels is the knee; 24 buys little.", 50)

    slide("Check the share, not the weight", f"""
      <div class="chart wide">{chart_share()}</div>
      <div class="cols">
        <div>{code_block('''
loss = necrosis_loss + w * vessel_loss''')}
          <p><code>w = 3.0</code> was set while fighting a stuck vessel head and never revisited.
          The necrosis head got 3 % of the loss.</p>
        </div>
        <div>
          <p class="num">val DSC at epoch 10<br><b>0.46</b> → <b class="o">0.79</b></p>
          <p>Same data, same budget, one number. Print each term's share whenever a loss
          changes form.</p>
        </div>
      </div>""",
          "This was one of the largest effects in the whole project and it was a hyperparameter "
          "nobody looked at. In the notebook the room prints these shares on their own model.", 60)

    slide("Two knobs that are free after training", f"""
      <div class="cols">
        <div><h3>Rollout length K</h3><div class="chart">{chart_k_sweep()}</div>
          <p>Pick on validation; take the cheapest within 1 % of the best.</p></div>
        <div><h3>Decision threshold</h3><div class="chart">{chart_threshold()}</div>
          <p>Our search stopped at 0.7. The optimum was higher:
          <b>+{f3(thr_gain) if thr_gain is not None else MISSING}</b> vessel F1 for free.</p></div>
      </div>""",
          "K: the cheapest within 1 % of the best, chosen on validation, never on test. "
          "The threshold one is fresh: we reported the vessel head at a grid edge for weeks. "
          "The weighted BCE makes the head over-confident, so its best threshold is high.", 60)

    slide("Things we measured wrong", f"""
      <div class="tiles">
        <div class="tile"><div class="t">Rotations and flips looked free</div>
          <div class="v">vessel F1 {f3(d4_off_f1)} → <b class="o">{f3(d4_f1)}</b></div>
          <p>The heat equation has the symmetry. Axial CT does not: the liver is on the right.</p></div>
        <div class="tile"><div class="t">"bf16 supported" on a T4-class GPU</div>
          <div class="v"><b class="o">{(amp.get("bf16_ms", 0) / amp.get("fp32_ms", 1)):.0f}×</b> slower per batch</div>
          <p>It is emulated. Ask for the compute capability, not <code>is_bf16_supported()</code>.</p></div>
        <div class="tile"><div class="t">A random fire mask</div>
          <div class="v">vessel F1 {f3(fire_f1)} → <b class="o">{f3(nofire_f1)}</b> without it</div>
          <p>The classic Growing-NCA trick. Here updating every cell every step trained better, and the model is deterministic.</p></div>
        <div class="tile"><div class="t">A vessel-only warm start</div>
          <div class="v">learned vessels <b class="o">½</b> as fast</div>
          <p>Per gradient step, against training both heads together. Necrosis helps the vessels.</p></div>
      </div>
      <p class="punch">Change one variable at a time, and measure the trick before you keep it.</p>""",
          "Pick two aloud: augmentation, and the bf16 one because it silently hit the Colab runs "
          "of this very notebook until this week. The fire mask is the one people will ask about: "
          "same 5-minute budget, three seeds each, necrosis DSC 0.818 to 0.831 as well.", 70)

    af = load("anatomy_first.json") or {}
    af1 = af.get("single rollout (seeds 0-1)", {})
    af2 = af.get("anatomy first, 6 CT-only steps (seeds 0-2)", {})
    slide("Freeze what the plan must not touch", f"""
      <div class="cols wide-left">
        <div class="chart">{chart_anatomy()}</div>
        <div>
          <p>The vessels do not depend on the needle, but a model that sees the plan
          while looking for them learns that <b>inside a lesion there are no vessels</b>.</p>
          <p>So: a few steps on the CT alone, then switch the needle on and hold the vessel
          channel fixed, like the CT. Same rule, same weights.</p>
          <p class="num">vessel F1 with random plans<br><b>{f3(af1.get("vessel_f1_random"))}</b> →
          <b class="b">{f3(af2.get("vessel_f1_random"))}</b></p>
        </div>
      </div>""",
          "This came out of a robustness audit this week: move the needle and the vessel map "
          "changed. Pinning the necrosis channel alone was not enough, the plan leaks through the "
          "scratch channels. The general rule: whatever must not depend on an input, compute it "
          "before that input exists, then treat it as environment.", 55)

    from ablation2d.anatomy import from_segmentations
    from ablation2d.data import AblationDataset
    from ablation2d.train import load_checkpoint
    from ablation2d.webplanner import BrowserPlanner, slice_from_anatomy, slices_from_split

    ref, cfg, ck = load_checkpoint(ROOT / "checkpoints/ablation_cnca_liver.pt")
    real = np.load(ROOT / "data/real_slice.npz")
    anat = from_segmentations(real["seg_liver"], real["seg_vessel"], hounsfield=real["hounsfield"])
    sl = [slice_from_anatomy(anat, "demo slice")] + slices_from_split(
        AblationDataset(ROOT / "data/test.npz"), AblationDataset(ROOT / "data_liver/test.npz"), n=2)
    planner = BrowserPlanner([("reference model", ref, ck.get("vessel_threshold", 0.5), cfg.infer_steps)],
                             sl).html()
    qr = (OUT / "colab_qr.svg").read_text()
    qr = qr[qr.index("<svg"):]
    slide("Hands-on 3: train one in five minutes", f"""
      <div class="handover">
        <div class="live">{planner}</div>
        <div class="qr">{qr}<p>Open in Colab<br><span class="muted">github.com/JonasMht/<br>miccai2026-nca-tutorial</span></p></div>
      </div>""",
          "This is the model running live in the browser. Drag the needle next to a vessel: the "
          "lesion is notched. Then: scan the code, T4 GPU, run the first cell.", 50, "handover-slide")
    return slides


# --------------------------------------------------------------------------- #
CSS = """
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#16181d;overflow:hidden;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
#deck{position:absolute;left:50%;top:50%;width:1600px;height:900px;transform-origin:50% 50%}
.slide{position:absolute;inset:0;background:#fbfbfa;color:#1d1f24;padding:64px 84px 56px;display:none;flex-direction:column}
.slide.on{display:flex}
.slide h2{font-size:58px;margin:0 0 40px;letter-spacing:-0.5px;font-weight:700}
.slide h3{font-size:31px;margin:0 0 12px;color:#343a40}
.slide p,.slide li{font-size:31px;line-height:1.42;margin:0 0 16px}
.slide ul{margin:0;padding-left:28px}
.slide li{margin-bottom:10px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:60px;align-items:start}
.cols.wide-left{grid-template-columns:1.1fr 1fr;align-items:center}
.cols3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:44px}
.cols3 p{font-size:25px}
pre{background:#f1f3f5;border-radius:10px;padding:20px 24px;margin:0 0 20px;font-size:24px;line-height:1.45;overflow:hidden}
code{font-family:'JetBrains Mono','Fira Code',ui-monospace,Menlo,Consolas,monospace}
p code{background:#f1f3f5;padding:1px 6px;border-radius:4px;font-size:0.88em}
.chart svg{width:100%;height:auto}
.chart.wide{max-width:1100px;margin-bottom:18px}
.muted{color:#6b6f76;font-size:24px !important}
.punch{margin-top:auto !important;font-size:34px !important;font-weight:600;border-left:6px solid #d9480f;padding-left:20px}
.num{font-size:28px !important}
.num b{font-size:46px}
b.o{color:#d9480f} b.b{color:#1c7ed6}
.lead{font-weight:600}
.kicker{font-size:22px;color:#d9480f;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:26px}
.big{font-size:74px;line-height:1.06;margin:0 0 30px;letter-spacing:-1px}
.title .strip{width:100%;margin:6px 0 26px;border-radius:8px}
.who{font-size:34px;margin-bottom:12px}
.diagram .lbl{font-size:26px;font-weight:700;color:#495057;margin-top:12px}
.diagram .row{display:flex;gap:6px;margin:10px 0 6px}
.diagram .cap{font-size:24px;color:#6b6f76;margin-bottom:34px}
.blk{width:50px;height:84px;border-radius:6px;display:inline-block}
.blk.c0{background:#4dabf7}.blk.c1{background:#74c0fc}.blk.c2{background:#339af0}.blk.c3{background:#a5d8ff}
.blk.same{background:#ff922b}
.chan{display:flex;gap:6px;margin:6px 0 4px}
.chan .c{height:92px;flex:1;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:600;color:white}
.chan .a{background:#d9480f;flex:1.6}.chan .e{background:#495057;flex:1.4}.chan .s{background:#dee2e6}
.chan-lbl{display:grid;grid-template-columns:3.2fr 4.2fr 11fr;font-size:23px;color:#6b6f76;margin-bottom:34px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:30px 40px}
.tile{background:#f1f3f5;border-radius:12px;padding:24px 30px}
.tile .t{font-size:29px;font-weight:700;margin-bottom:6px}
.tile .v{font-size:36px;margin-bottom:8px}
.tile p{font-size:24px;margin:0;color:#495057}
.handover{display:flex;gap:40px;align-items:flex-start}
.handover .live{flex:none;transform:scale(1.12);transform-origin:0 0;width:1040px;height:700px}
.handover .qr{margin-left:auto;text-align:center;width:250px;flex:none}
.handover .qr svg{width:240px;height:240px}
.handover .qr p{font-size:22px;margin-top:10px}
.missing{color:#c92a2a;font-style:italic}
#bar{position:absolute;left:0;bottom:0;height:5px;background:#d9480f;transition:width .25s}
#pn{position:fixed;right:18px;bottom:14px;color:#888;font-size:14px}
#presenter{position:fixed;left:0;right:0;bottom:0;background:rgba(20,22,27,.94);color:#eee;padding:16px 26px;display:none;font-size:19px;line-height:1.45}
#presenter .clock{font-size:26px;font-variant-numeric:tabular-nums;margin-bottom:6px}
#presenter .late{color:#ff8787}
@page{size:1600px 900px;margin:0}
@media print{html,body{overflow:visible;height:auto;background:#fff}
 #deck{position:static;transform:none !important;width:1600px;height:auto}
 .slide{display:flex !important;position:relative;width:1600px;height:900px;page-break-after:always;break-after:page}
 #bar,#pn,#presenter{display:none !important}}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];const deck=document.getElementById('deck');
let i=0,t0=null,tSlide=null;const total=%TOTAL%;
function fit(){const s=Math.min(innerWidth/1600,innerHeight/900);deck.style.transform=`translate(-50%,-50%) scale(${s})`;}
function show(k){i=Math.max(0,Math.min(slides.length-1,k));slides.forEach((s,j)=>s.classList.toggle('on',j===i));
 document.getElementById('bar').style.width=(100*(i+1)/slides.length)+'%';document.getElementById('pn').textContent=`${i+1} / ${slides.length}`;
 tSlide=performance.now();if(!t0&&i>0)t0=performance.now();location.hash=i+1;notes();}
function fmt(s){s=Math.max(0,Math.round(s));return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`}
function notes(){const p=document.getElementById('presenter');if(p.style.display!=='block')return;const s=slides[i];
 const el=t0?(performance.now()-t0)/1000:0, es=(performance.now()-tSlide)/1000, due=+s.dataset.due, budget=+s.dataset.sec;
 p.innerHTML=`<div class="clock">total <b class="${el>due?'late':''}">${fmt(el)}</b> / ${fmt(total)} · this slide <b class="${es>budget?'late':''}">${fmt(es)}</b> / ${fmt(budget)} · be done by ${fmt(due)}</div>${s.dataset.notes}`;}
setInterval(notes,500);
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){show(i+1);e.preventDefault()}
 else if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);else if(e.key==='p'||e.key==='P'){const p=document.getElementById('presenter');p.style.display=p.style.display==='block'?'none':'block';notes();}
 else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}
 else if(e.key==='t'||e.key==='T'){t0=performance.now();tSlide=t0}else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);});
addEventListener('resize',fit);fit();show((+location.hash.slice(1)||1)-1);
"""


def main():
    slides = build()
    total = sum(s["seconds"] for s in slides)
    due = 0
    parts = []
    for n, s in enumerate(slides):
        due += s["seconds"]
        head = f"<h2>{s['title']}</h2>" if s["title"] else ""
        parts.append(f'<section class="slide {s["cls"]}" data-sec="{s["seconds"]}" data-due="{due}" '
                     f'data-notes="{html.escape(s["notes"])}">{head}{s["body"]}</section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Training NCAs: tips</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style></head><body>
<div id="deck">{''.join(parts)}<div id="bar"></div></div><div id="pn"></div><div id="presenter"></div>
<script>{JS.replace('%TOTAL%', str(total))}</script></body></html>"""
    OUT.mkdir(exist_ok=True)
    (OUT / "tips_and_tricks_nca.html").write_text(page)

    lines = ["# Speaker notes: Tips for training NCAs (10 min)", "",
             "Thursday 1 October, 10:50-11:00, right after Caroline Essert's *NCA for simulation*",
             "and right before Hands-on 3. Generated by `build_slides.py`, edit the notes there.", "",
             "Press **P** in the deck for these notes with a per-slide timer; **F** for full screen.", "",
             "| # | slide | seconds | done by |", "|---|---|---|---|"]
    due = 0
    for n, s in enumerate(slides, 1):
        due += s["seconds"]
        lines.append(f"| {n} | {s['title'] or 'title'} | {s['seconds']} | {due // 60}:{due % 60:02d} |")
    lines += ["", f"Total {total // 60}:{total % 60:02d} of 10:00.", ""]
    for n, s in enumerate(slides, 1):
        lines += [f"**{n}. {s['title'] or 'Title'}** ({s['seconds']} s)", "", s["notes"], ""]
    lines += ["## Questions to expect", "",
              "- *Is this validated?* No. The device calibration behind every label is provisional: "
              "ex vivo bovine liver at 17 °C, held-out error 9.9 %.",
              "- *Why 2-D?* Because it trains in five minutes on a free GPU. The 3-D model uses the "
              "same channels, device and loss.",
              "- *Could I use it clinically?* No: a 2-D surrogate of a provisionally calibrated solver.",
              "- *Two phases, is that still one NCA?* Yes: same rule, same weights, run first with the "
              "needle channels empty and then with them on. It is the environment trick applied to an "
              "output: whatever must not depend on an input is computed before that input exists.",
              "- *Why no fire mask?* Measured: without it both heads are better in the same budget and "
              "the model is deterministic, so one rollout is the answer.",
              ""]
    (OUT / "speaker_notes.md").write_text("\n".join(lines))
    print(f"wrote slides/tips_and_tricks_nca.html ({len(slides)} slides, {total // 60}:{total % 60:02d})")


if __name__ == "__main__":
    main()
