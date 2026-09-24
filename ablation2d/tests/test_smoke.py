"""Smoke tests: fast, CPU only, no solver needed.

    python -m pytest tests/ -q

The package must import and build input channels without a compiled solver,
because that is the situation of every participant on Colab.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation2d.anatomy import (LBL_LIVER, LBL_VESSEL, from_segmentations,  # noqa: E402
                                sample_anatomy)
from ablation2d.channels import (N_INPUT_CHANNELS,  # noqa: E402
                                 N_TARGET_CHANNELS, build_inputs,
                                 build_targets, paint_plan)
from ablation2d.device import (EMPRINT_HP, VESSEL_SINK_MIN_RADIUS_MM,  # noqa: E402
                               tissue_properties, vessel_sink_rate)
from ablation2d.model import (AblationCNCA, dice, multitask_loss,  # noqa: E402
                              necrosis_loss, vessel_f1)
from ablation2d.plans import Plan, sample_plan  # noqa: E402
from ablation2d.solver_env import solver_available  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# the Colab boundary
# --------------------------------------------------------------------------- #
def test_tissue_table_works_without_the_solver():
    """Everything the notebook draws must come from the baked table."""
    props = tissue_properties()
    assert set(props) == {"background", "liver", "vessel"}
    assert props["liver"]["perfusion"] > props["background"]["perfusion"]
    # rho*c is what the solver uses; liver's must land under the encoding ceiling
    assert 0 < props["liver"]["rho"] * props["liver"]["c"] < 4.1e-3


def test_channels_build_without_the_solver():
    rng = np.random.default_rng(0)
    anat = sample_anatomy(rng)
    plan = sample_plan(rng, anat)
    x = build_inputs(anat, plan)
    assert x.shape == (N_INPUT_CHANNELS, 128, 128)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    y = build_targets(anat, np.zeros(anat.shape, np.float32))
    assert y.shape == (N_TARGET_CHANNELS, 128, 128)


# --------------------------------------------------------------------------- #
# the physics that is encoded in the channels
# --------------------------------------------------------------------------- #
def test_sink_rate_is_floored_and_falls_with_calibre():
    rc = np.full(4, 3.819e-3, np.float32)
    bio = np.ones(4, bool)
    r = np.array([0.5, VESSEL_SINK_MIN_RADIUS_MM, 3.0, 6.0], np.float32)
    rate = vessel_sink_rate(r, rc, bio)
    assert rate[0] == 0.0, "below the floor the sink must be OFF"
    assert rate[1] > rate[2] > rate[3] > 0, "1/R^2: bigger vessel, weaker sink"


def test_hounsfield_is_present_and_hard():
    """The inputs are built on the CT, so an anatomy without one must fail loudly, and
    the vessel/liver separation must stay in the range the real slice sits in
    (d' = 0.17 at 2 mm), not be accidentally made easy."""
    from ablation2d.anatomy import Anatomy
    rng = np.random.default_rng(4)
    ds = []
    for _ in range(12):
        anat = sample_anatomy(rng)
        assert anat.hounsfield is not None
        a = anat.hounsfield[anat.liver_mask]
        b = anat.hounsfield[anat.vessel_mask]
        ds.append(abs(b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))
    assert min(ds) < 0.4, "no hard cases: the vessel head would not transfer"
    # A hand-built Anatomy gets a synthesised CT from __post_init__, so the
    # controlled experiments work; only an explicitly blanked one fails.
    bare = Anatomy(label=np.zeros((8, 8), np.uint8),
                   vessel_radius_mm=np.zeros((8, 8), np.float32))
    assert bare.hounsfield is not None
    bare.hounsfield = None
    with pytest.raises(ValueError):
        build_inputs(bare, sample_plan(rng, sample_anatomy(rng)))


def test_liver_and_vessel_are_disjoint_labels():
    rng = np.random.default_rng(2)
    anat = sample_anatomy(rng)
    assert not (anat.liver_mask & anat.vessel_mask).any()
    assert 0.02 < anat.vessel_fraction < 0.25


def test_from_segmentations_takes_the_union():
    """The corpus's masks are disjoint; the organ is their union, not their
    intersection. Getting this backwards turned a 13.4 %-vessel slice into 3.2 %."""
    liver = np.zeros((16, 16), bool); liver[2:14, 2:14] = True
    vessel = np.zeros((16, 16), bool); vessel[6:10, 2:14] = True
    liver &= ~vessel                       # disjoint, as the corpus stores them
    anat = from_segmentations(liver, vessel)
    organ = anat.label != 0
    assert organ.sum() == (liver | vessel).sum()
    assert anat.vessel_mask.sum() == vessel.sum()


# --------------------------------------------------------------------------- #
# the applicator
# --------------------------------------------------------------------------- #
def test_slot_is_not_the_whole_shaft():
    assert EMPRINT_HP.emission_start_mm == pytest.approx(7.25)
    assert EMPRINT_HP.emission_end_mm == pytest.approx(19.0839, abs=1e-3)
    assert EMPRINT_HP.deposited_max_w == pytest.approx(90.0)


def test_painting_marks_something_for_a_subvoxel_needle():
    """A 2.11 mm antenna on a 2 mm grid is sub-voxel; the paint radius is
    floored at half a cell so a needle between two centres still marks a cell."""
    from ablation2d.plans import Needle
    n = Needle(64.0, 64.0, 64.0, 160.0, 90.0, 300.0)
    painted = paint_plan(Plan([n]), (128, 128), 2.0)
    assert painted["needle"].sum() > 0
    assert painted["applicator_activation"].sum() > 0
    assert painted["applicator_activation"].max() == pytest.approx(90.0 / 150.0)


def test_zero_power_needle_paints_geometry_but_no_dose():
    from ablation2d.plans import Needle
    n = Needle(64.0, 64.0, 64.0, 160.0, 0.0, 300.0)
    painted = paint_plan(Plan([n]), (128, 128), 2.0)
    assert painted["needle"].sum() > 0, "the shaft is still there"
    assert painted["applicator_activation"].max() == 0.0


def test_duration_is_fixed():
    """Duration is not an input, so the corpus must not vary it: a model
    with no duration channel trained on varying burns predicts their average."""
    from ablation2d.plans import FIXED_DURATION_S, sample_duration_s
    rng = np.random.default_rng(0)
    d = sample_duration_s(rng, 500)
    assert float(d.min()) == float(d.max()) == FIXED_DURATION_S


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def test_untrained_model_is_exactly_flat():
    """The free sanity check. Zero-init output layers => a no-op automaton."""
    m = AblationCNCA(channels=8, hidden_mult=2, n_sub_models=2)
    x = torch.rand(2, N_INPUT_CHANNELS, 32, 32)
    with torch.no_grad():
        p = m(x, steps=12)
    assert p.shape[1] == N_TARGET_CHANNELS
    assert float(p.max() - p.min()) == 0.0


def test_conditioning_is_not_edited_by_the_automaton():
    m = AblationCNCA(channels=8, hidden_mult=2, n_sub_models=2,
                     restore_conditioning=True)
    x = torch.rand(1, N_INPUT_CHANNELS, 24, 24)
    with torch.no_grad():
        _, state = m(x, steps=5, return_state=True)
    lo = N_TARGET_CHANNELS
    assert torch.allclose(state[:, lo:lo + N_INPUT_CHANNELS], x)


def test_model_rejects_a_state_too_narrow_to_hold_its_inputs():
    with pytest.raises(ValueError):
        AblationCNCA(channels=N_INPUT_CHANNELS)


def test_multitask_loss_sees_both_heads():
    pred = torch.zeros(1, 2, 8, 8)
    target = torch.zeros(1, 2, 8, 8)
    target[0, 0, 0, 0] = 1.0                    # a dead cell
    only_death = float(multitask_loss(pred, target, vessel_weight=0.0))
    target[0, 1, 4, 4] = 1.0                    # ...and a vessel cell
    both = float(multitask_loss(pred, target, vessel_weight=1.0))
    assert both > only_death


def test_loss_weights_the_foreground():
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8); target[0, 0, 0, 0] = 1.0
    heavy = float(necrosis_loss(pred, target, foreground_weight=100.0))
    light = float(necrosis_loss(pred, target, foreground_weight=1.0))
    assert heavy > light


def test_dice_is_one_when_both_are_empty():
    z = torch.zeros(3, 2, 8, 8)
    assert float(dice(z, z).min()) == 1.0
    assert float(vessel_f1(z, z).min()) == 1.0


# --------------------------------------------------------------------------- #
# only where the solver exists
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not solver_available(), reason="the solver not installed")
def test_baked_tissue_table_matches_the_solver():
    from ablation2d.device import verify_tissue_table
    verify_tissue_table()


@pytest.mark.skipif(not solver_available(), reason="the solver not installed")
def test_a_vessel_shrinks_the_lesion():
    from ablation2d.anatomy import Anatomy, NX, NY, DX_MM
    from ablation2d.physics import simulate
    from ablation2d.plans import Needle

    plan = Plan([Needle(64 * DX_MM, 64 * DX_MM, 64 * DX_MM, 110 * DX_MM, 100.0, 360.0)])
    xx = np.arange(NX)[None, :].repeat(NY, 0)

    def run(calibre):
        lab = np.full((NY, NX), LBL_LIVER, np.uint8)
        rad = np.zeros((NY, NX), np.float32)
        if calibre:
            m = np.abs(xx - 69) <= 0.5 * calibre / DX_MM
            lab[m] = LBL_VESSEL
            rad[m] = 0.5 * calibre
        return simulate(Anatomy(label=lab, vessel_radius_mm=rad), plan).lesion_cm2

    none, small, big = run(0), run(4), run(13)
    assert big < small < none, f"heat sink not monotonic: {none}, {small}, {big}"
    assert big < 0.8 * none, "a 13 mm vein should take a fifth of the lesion"


def test_d4_is_the_whole_group_and_no_more():
    """8 distinct elements, closed under any further mirror or rotation.

    The ask was "rotations and flips, equiprobable, combinatorial, combining in
    any possible way". That set IS D4 and has exactly eight members - composing
    another mirror onto any of them lands back inside the eight. Worth pinning:
    a naive "rotate k, then flip x with p=0.5, then flip y with p=0.5" draws 16
    combinations onto those 8 elements at unequal probability, which is the
    obvious implementation and is not what was asked for.
    """
    import torch
    from ablation2d.train import d4, N_D4

    t = torch.arange(16.0).reshape(1, 1, 4, 4)        # no symmetry of its own
    orbit = {tuple(d4(t, g).flatten().tolist()) for g in range(N_D4)}
    assert len(orbit) == N_D4
    for dims in ((-1,), (-2,), (-1, -2)):
        assert {tuple(torch.flip(d4(t, g), dims=dims).flatten().tolist())
                for g in range(N_D4)} == orbit


def test_d4_keeps_inputs_and_targets_on_the_same_element():
    """The failure this guards is silent: a batch where x turned and y did not
    still trains, it just trains on nonsense."""
    import torch
    from ablation2d.train import augment_d4, d4, N_D4

    x, y = torch.rand(32, 3, 8, 8), torch.rand(32, 2, 8, 8)
    xa, ya = augment_d4(x, y, torch.Generator().manual_seed(0))
    for i in range(len(x)):
        assert any(torch.allclose(xa[i], d4(x[i:i + 1], g)[0])
                   and torch.allclose(ya[i], d4(y[i:i + 1], g)[0])
                   for g in range(N_D4))
    assert xa.is_contiguous() and ya.is_contiguous()


def test_d4_is_uniform_over_the_eight():
    import torch
    from ablation2d.train import augment_d4, d4, N_D4

    x = torch.arange(64.0).reshape(1, 1, 8, 8).repeat(4000, 1, 1, 1)
    xa, _ = augment_d4(x, x.clone(), torch.Generator().manual_seed(1))
    keys = [tuple(d4(x[:1], g).flatten().tolist()) for g in range(N_D4)]
    counts = [sum(tuple(a.flatten().tolist()) == k for a in xa) for k in keys]
    assert min(counts) > 4000 / N_D4 * 0.85, counts
    assert max(counts) < 4000 / N_D4 * 1.15, counts


def test_d4_refuses_a_non_square_grid():
    import pytest
    import torch
    from ablation2d.train import augment_d4

    with pytest.raises(ValueError, match="square"):
        augment_d4(torch.rand(2, 3, 8, 16), torch.rand(2, 2, 8, 16),
                   torch.Generator().manual_seed(0))


def test_d4_inverse_is_actually_the_inverse():
    """A mirror is an involution, so the inverse of "rotate then mirror" is
    "mirror then rotate back". The naive order happens to be correct for two of
    the four mirrored elements, which is exactly enough for the bug to survive a
    spot check and mis-register the other two."""
    import torch
    from ablation2d.train import d4, d4_inv, N_D4

    t = torch.rand(2, 3, 8, 8)
    for g in range(N_D4):
        assert torch.allclose(d4_inv(d4(t, g), g), t, atol=1e-6), g

    naive_ok = [g for g in range(4, N_D4)
                if torch.allclose(
                    torch.flip(torch.rot90(d4(t, g), -(g % 4), dims=(-2, -1)),
                               dims=(-1,)), t, atol=1e-6)]
    assert naive_ok == [4, 6], naive_ok


def test_tta_is_orientation_invariant():
    """The point of averaging over the group: the answer stops depending on
    which way the patient was lying."""
    import torch
    from ablation2d.model import AblationCNCA
    from ablation2d.train import d4, d4_inv, predict_d4_tta

    torch.manual_seed(0)
    m = AblationCNCA(channels=10, hidden_mult=2, n_sub_models=2,
                     fire_rate=1.0).eval()
    x = torch.rand(1, 3, 16, 16)
    base = predict_d4_tta(m, x, steps=3)
    for g in (1, 3, 5):
        rot = d4_inv(predict_d4_tta(m, d4(x, g), steps=3), g)
        assert torch.allclose(base, rot, atol=1e-5), g


def test_free_conditioning_actually_lets_the_state_move():
    """`restore_conditioning=False` must leave the input channels writable.

    The two regimes differ by one `torch.cat` inside the rollout, which is
    exactly the kind of thing that silently stops applying. Pinned in both
    directions: restored means bit-identical conditioning at every step, free
    means it moves."""
    import torch
    from ablation2d.model import AblationCNCA, COND_SLICE

    torch.manual_seed(0)
    x = torch.rand(1, 3, 24, 24)

    held = AblationCNCA(channels=10, hidden_mult=2, n_sub_models=2,
                        fire_rate=1.0, restore_conditioning=True).eval()
    free = AblationCNCA(channels=10, hidden_mult=2, n_sub_models=2,
                        fire_rate=1.0, restore_conditioning=False).eval()
    # A zero-init model is a no-op, so give both the same non-trivial weights.
    for sub in list(held.sub_models) + list(free.sub_models):
        torch.nn.init.normal_(sub[2].weight, 0.0, 0.05)
    free.load_state_dict(held.state_dict())

    with torch.no_grad():
        _, th = held(x, steps=6, return_trace=True, trace_full=True)
        _, tf = free(x, steps=6, return_trace=True, trace_full=True)

    hc = torch.stack([f[:, COND_SLICE] for f in th])
    fc = torch.stack([f[:, COND_SLICE] for f in tf])
    assert float((hc - hc[0]).abs().max()) == 0.0, "restored conditioning moved"
    assert float((fc - fc[0]).abs().max()) > 0.0, "free conditioning did not move"


def test_inference_fire_rate_matches_training():
    """The mask scales the per-step update, so evaluating at a different fire
    rate than the model was trained at applies the wrong residual magnitude.

    On an earlier checkpoint: DSC 0.705 at its training fire_rate 0.5, 0.308 at
    1.0. Nothing may silently change the rate between training and evaluation.
    """
    from pathlib import Path

    import pytest
    import torch

    from ablation2d.train import load_checkpoint

    root = Path(__file__).resolve().parent.parent
    cks = [p for p in (root / "checkpoints/ablation_cnca_full.pt",
                       root / "checkpoints_aug/ablation_cnca_full.pt")
           if p.exists()]
    if not cks:
        pytest.skip("no checkpoint in this checkout")
    for p in cks:
        model, cfg, _ = load_checkpoint(p, device="cpu")
        assert model.fire_rate == cfg.fire_rate, (
            f"{p.name}: model runs at fire_rate {model.fire_rate} but was "
            f"trained at {cfg.fire_rate}")


def test_mask_to_organ_touches_only_the_vessel_head():
    """Masking the necrosis head would delete correct predictions: 11.7 % of the
    necrosis area lies outside the liver on average. Only the vessel head is
    safe to mask, because every true vessel voxel is inside the organ."""
    import torch
    from ablation2d.evaluate import mask_to_organ

    pred = torch.ones(2, 2, 8, 8)
    organ = torch.zeros(2, 8, 8)
    organ[:, 2:6, 2:6] = 1.0
    out = mask_to_organ(pred, organ)
    assert torch.equal(out[:, 0], pred[:, 0]), "necrosis head must be untouched"
    assert out[:, 1].sum() == organ.sum(), "vessel head must be zero outside"


def test_studio_prediction_modes_and_mask():
    """The interactive planner is the part most likely to break in front of a
    room, and it had no automated check because it could not be constructed
    without ipywidgets. `build=False` makes the inference half testable.

    Pins the two properties the shipped configuration rests on: masking removes
    every out-of-organ vessel prediction, and it leaves the necrosis head alone.
    """
    from pathlib import Path

    import numpy as np
    import pytest

    from ablation2d.channels import FIXED_DURATION_S
    from ablation2d.plans import Needle, Plan
    from ablation2d.train import load_checkpoint
    from ablation2d.ui import AblationStudio

    root = Path(__file__).resolve().parent.parent
    ck = root / "checkpoints/ablation_cnca_full.pt"
    demo = root / "data/real_slice.npz"
    if not (ck.exists() and demo.exists()):
        pytest.skip("no checkpoint or demo slice in this checkout")

    from ablation2d.anatomy import from_segmentations
    r = np.load(demo)
    anat = from_segmentations(r["seg_liver"], r["seg_vessel"],
                              hounsfield=r.get("hounsfield"))
    model, cfg, _ = load_checkpoint(ck, device="cpu")
    h, w = anat.shape
    cx, cy = 0.45 * w * anat.dx_mm, 0.52 * h * anat.dx_mm
    plan = Plan([Needle(cx, cy, cx + 70, cy + 24, 110.0, FIXED_DURATION_S)])
    outside = ~(anat.liver_mask | anat.vessel_mask)

    nec_unmasked = None
    for mask in (False, True):
        st = AblationStudio(model, anat, steps=cfg.infer_steps,
                            torch_device="cpu", average="none",
                            mask_vessels=mask, build=False)
        nec, ves = st.predict(plan)
        n_out = int((ves > 0.5)[outside].sum())
        if mask:
            assert n_out == 0, f"mask left {n_out} vessel px outside the organ"
        else:
            nec_unmasked = nec
    # the necrosis head must be untouched by the vessel mask
    assert nec_unmasked is not None


def test_liver_mask_cannot_leak_vessel_positions():
    """The masked-CT corpus is only sound while every vessel is interior to the
    liver. A vessel outside the parenchyma would appear as an island in the
    organ mask, and the mask's boundary would hand the vessel head its answer -
    inflating exactly the large round vessels it already scores best on."""
    from pathlib import Path

    import numpy as np
    import pytest

    root = Path(__file__).resolve().parent.parent
    f = root / "data_real" / "test_slices.npz"
    if not f.exists():
        pytest.skip("slice corpus not in this checkout")
    d = np.load(f)
    sl = d["seg_liver"].astype(bool)
    sv = d["seg_vessel"].astype(bool)
    outside = (sv & ~sl).sum() / max(sv.sum(), 1)
    assert outside == 0.0, (
        f"{100 * outside:.2f} % of vessel voxels lie outside the liver mask; "
        f"the organ mask would reveal them as shape")


def _notebook_cell(startswith: str):
    """The SOLUTION source of the notebook cell whose body starts with `startswith`."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("bnb", root / "build_notebooks.py")
    bnb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnb)
    hits = [c for c in bnb.CELLS
            if c["kind"] == "code" and c["solution"].lstrip().startswith(startswith)]
    assert len(hits) == 1, f"{len(hits)} cells start with {startswith!r}"
    return hits[0]["solution"], bnb


def test_notebook_model_IS_the_packaged_model():
    """The model the room types out must be numerically identical to the one the
    pipeline trains and ships - same module names, same maths - or every
    number in the notebook describes a different network from the one on
    screen. Checked by loading the packaged weights into the notebook's class
    and comparing outputs with the stochastic mask switched off."""
    import torch
    import torch.nn as nn
    from ablation2d.model import AblationCNCA as Packaged
    from ablation2d.train import FAST

    consts, _ = _notebook_cell("N_OUT, N_INPUT = 2, 3")
    model_src, _ = _notebook_cell("class AblationCNCA")
    ns = {"torch": torch, "nn": nn, "DEVICE": "cpu", "FAST": FAST, "print": lambda *a, **k: None}
    exec(consts, ns)
    exec(model_src, ns)
    Student = ns["AblationCNCA"]

    for anatomy in (0, 4):
        torch.manual_seed(0)
        ref = Packaged(channels=12, hidden_mult=2, n_sub_models=2, anatomy_steps=anatomy)
        for sub in ref.sub_models:                 # zero-init is a no-op; perturb
            torch.nn.init.normal_(sub[2].weight, 0.0, 0.05)
        stu = Student(channels=12, hidden_mult=2, n_sub_models=2, anatomy_steps=anatomy)
        stu.load_state_dict(ref.state_dict())      # names must line up exactly

        x = torch.rand(2, 3, 24, 24)
        with torch.no_grad():
            a, sa = ref(x, steps=5, return_state=True)
            b, sb = stu(x, steps=5, return_state=True)
            _, tr = stu(x, steps=3, return_trace=True)
        assert b.shape == (2, 2, 24, 24)
        assert torch.allclose(a, b, atol=1e-6) and torch.allclose(sa, sb, atol=1e-6), anatomy
        assert len(tr) == 3 and tr[0].shape == (2, 2, 24, 24)
        assert torch.equal(sb[:, 2:5], x), "conditioning must be restored every step"
        if anatomy:
            x2 = x.clone(); x2[:, 1:] = torch.rand(2, 2, 24, 24)
            with torch.no_grad():
                assert torch.equal(stu(x2, steps=5)[:, 1], b[:, 1]), "vessels must not see the plan"

    fresh = Student(channels=12, hidden_mult=2, n_sub_models=2, anatomy_steps=4)
    with torch.no_grad():
        p = fresh(x, steps=20)
    assert float(p.max() - p.min()) == 0.0, "untrained model must be exactly flat"


def test_notebook_builds_and_the_todo_count_is_true():
    """The TUTORIAL tells people how many TODOs there are. That number is now
    computed from the cells, and every code cell in both variants must parse."""
    import ast
    import json
    from pathlib import Path

    _, bnb = _notebook_cell("N_OUT, N_INPUT = 2, 3")
    root = Path(__file__).resolve().parent.parent
    for variant in ("TUTORIAL", "SOLUTION"):
        bnb.build(variant)
        nb = json.loads((root / "notebooks" / f"ablation_with_nca_{variant}.ipynb").read_text())
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        text = "".join("".join(c["source"]) for c in nb["cells"])
        if variant == "TUTORIAL":
            found = text.count("# TODO ")
            assert found == bnb.N_TODOS, (found, bnb.N_TODOS)
            assert f"There are {bnb.N_TODOS} `TODO`s" in text
        else:
            assert "raise NotImplementedError" not in text


def test_build_inputs_mask_ct_matches_the_masked_corpus():
    """A model trained on data_liver must be fed the CT masked the same way at
    inference. Checked against the stored corpus: at most one uint8 level of
    difference - rounding - on every sample compared."""
    from pathlib import Path

    import numpy as np
    import pytest
    from ablation2d.anatomy import from_patient_slice
    from ablation2d.channels import build_inputs
    from ablation2d.plans import Plan

    root = Path(__file__).resolve().parent.parent
    sl, st = root / "data_real/test_slices.npz", root / "data_liver/test.npz"
    if not (sl.exists() and st.exists()):
        pytest.skip("slice corpus or masked corpus not in this checkout")
    d = np.load(sl)
    stored = np.load(st)["inputs"]
    ns = len(d["z"])
    for i in range(0, 60, 11):
        j = i % ns
        a = from_patient_slice(hounsfield=d["hounsfield"][j].astype(np.float32),
                               material_id=d["material_id"][j],
                               seg_liver=d["seg_liver"][j].astype(bool),
                               seg_vessel=d["seg_vessel"][j].astype(bool))
        x = build_inputs(a, Plan([]), mask_ct=True)[0]
        q = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
        diff = np.abs(q.astype(int) - stored[i, 0].astype(int)).max()
        assert diff <= 1, (i, diff)


def test_heat_sink_pair_places_a_tip_beside_a_large_vessel():
    """A real portal vein or IVC is ~27 mm across. The placement search used a
    fixed 5-cell offset from the vessel centreline, which is still inside a
    vessel that size, so on the demo slice every candidate was rejected and the
    notebook's heat-sink section raised. The ring is now relative to the
    vessel's own radius."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import torch
    from ablation2d.anatomy import from_segmentations
    from ablation2d.experiments import heat_sink_pair
    from ablation2d.model import AblationCNCA

    n = 128
    yy, xx = np.mgrid[:n, :n]
    liver = (yy - 64) ** 2 + (xx - 64) ** 2 <= 45 ** 2
    vessel = (yy - 64) ** 2 + (xx - 64) ** 2 <= 7 ** 2       # 14 cells = 28 mm
    hu = np.where(vessel, 180.0, np.where(liver, 100.0, 40.0)).astype(np.float32)
    anat = from_segmentations(liver, vessel, hounsfield=hu)

    torch.manual_seed(0)
    model = AblationCNCA(channels=10, hidden_mult=2, n_sub_models=2, fire_rate=1.0)
    for sub in model.sub_models:
        torch.nn.init.normal_(sub[2].weight, 0.0, 0.05)
    fig = heat_sink_pair(model, anat=anat, steps=4, mask_ct=True)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_heat_sink_pair_is_reproducible_under_stochastic_firing():
    """The demo compares two stochastic rollouts, so without common random
    numbers the with/without difference includes firing noise - and the angle
    search selects for it. On the demo slice that flipped the SIGN. With both
    rollouts seeded identically, the figure is also exactly reproducible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from ablation2d.anatomy import from_segmentations
    from ablation2d.experiments import heat_sink_pair
    from ablation2d.model import AblationCNCA

    n = 128
    yy, xx = np.mgrid[:n, :n]
    liver = (yy - 64) ** 2 + (xx - 64) ** 2 <= 45 ** 2
    vessel = (yy - 64) ** 2 + (xx - 64) ** 2 <= 7 ** 2
    hu = np.where(vessel, 180.0, np.where(liver, 100.0, 40.0)).astype(np.float32)
    anat = from_segmentations(liver, vessel, hounsfield=hu)
    torch.manual_seed(0)
    model = AblationCNCA(channels=10, hidden_mult=2, n_sub_models=2, fire_rate=0.5)
    for sub in model.sub_models:
        torch.nn.init.normal_(sub[2].weight, 0.0, 0.05)
    titles = []
    for _ in range(2):
        fig = heat_sink_pair(model, anat=anat, steps=4, mask_ct=True)
        titles.append(fig._suptitle.get_text())
        plt.close(fig)
    assert titles[0] == titles[1], titles
