"""The browser planner must compute what the PyTorch model computes.

Runs the page in headless Chromium (software WebGL), paints a plan, runs the
automaton with firing made deterministic (fire_rate = 1) and compares the whole
state with the PyTorch rollout. Skipped when Playwright is not installed.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation2d.anatomy import from_segmentations          # noqa: E402
from ablation2d.channels import build_inputs                # noqa: E402
from ablation2d.plans import FIXED_DURATION_S, Needle, Plan  # noqa: E402
from ablation2d.train import load_checkpoint                # noqa: E402
from ablation2d.webplanner import BrowserPlanner, slice_from_anatomy  # noqa: E402

playwright = pytest.importorskip("playwright.sync_api")

NEEDLES = [{"x": 120.0, "y": 118.0, "angle": 1.9, "power": 90},
           {"x": 150.0, "y": 140.0, "angle": 0.3, "power": 45}]


def _base(n, fov):
    u = np.array([math.cos(n["angle"]), math.sin(n["angle"])])
    t = 120.0
    for o, d in ((n["x"], u[0]), (n["y"], u[1])):
        if abs(d) > 1e-9:
            t = min(t, ((fov if d > 0 else 0.0) - o) / d)
    return np.array([n["x"], n["y"]]) + u * max(t, 25.0)


@pytest.fixture(scope="module")
def page_and_model(tmp_path_factory):
    model, cfg, _ = load_checkpoint(ROOT / "checkpoints" / "ablation_cnca_liver.pt")
    real = np.load(ROOT / "data" / "real_slice.npz")
    anat = from_segmentations(real["seg_liver"], real["seg_vessel"], hounsfield=real["hounsfield"])
    html = BrowserPlanner({"reference": model}, [slice_from_anatomy(anat)]).html()
    path = tmp_path_factory.mktemp("web") / "planner.html"
    path.write_text(f"<html><body>{html}</body></html>")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(path.as_uri())
        page.wait_for_timeout(500)
        assert not errors, errors
        assert page.evaluate("!!document.querySelector('[id^=nca-planner]').nca"), \
            "planner did not start (no WebGL2?)"
        yield page, model, anat
        browser.close()


def _call(page, js):
    return page.evaluate(f"(() => {{ const r = document.querySelector('[id^=nca-planner]'); {js} }})()")


def test_inputs_match_python(page_and_model):
    page, model, anat = page_and_model
    _call(page, f"r.nca.pause(); r.nca.setNeedles({NEEDLES});")
    js = np.array(_call(page, "return r.nca.input();"), np.float32).reshape(128, 128, 4)
    fov = 128 * anat.dx_mm
    plan = Plan([Needle(n["x"], n["y"], *_base(n, fov), n["power"], FIXED_DURATION_S)
                 for n in NEEDLES])
    py = build_inputs(anat, plan, mask_ct=True)
    for c, name in enumerate(("CT", "needle", "power")):
        assert np.abs(js[..., c] - py[c]).max() <= 1.0 / 255 + 1e-6, name


def _state(page, k, channels):
    js = np.array(_call(page, f"return r.nca.run({k}, 1.0);"), np.float32)
    t = -(-channels // 4)
    return js.reshape(128, t, 128, 4).transpose(1, 3, 0, 2).reshape(t * 4, 128, 128)[:channels]


def test_rollout_matches_pytorch(page_and_model):
    """Float32 on two different GPUs agrees to ~1e-4 per step. With firing
    forced to 1 the model is outside its training regime and amplifies that
    noise, so the state is compared exactly over 3 steps and by its
    thresholded necrosis after 12."""
    page, model, anat = page_and_model
    _call(page, f"r.nca.pause(); r.nca.setNeedles({NEEDLES});")
    inp = np.array(_call(page, "return r.nca.input();"), np.float32).reshape(128, 128, 4)
    x = torch.from_numpy(inp[..., :3].transpose(2, 0, 1).copy())[None]
    model.fire_rate = 1.0

    js3 = _state(page, 3, model.channels)
    js12 = _state(page, 9, model.channels)
    with torch.no_grad():
        _, s3 = model(x, steps=3, return_state=True)
        _, s12 = model(x, steps=9, state=s3, return_state=True)
    assert np.abs(js3 - s3[0].numpy()).max() < 3e-3
    differ = ((js12[0] > 0) != (s12[0, 0].numpy() > 0)).mean()
    assert differ < 0.002, f"{100 * differ:.2f} % of necrosis pixels differ"


def test_two_phase_rollout_matches_pytorch(page_and_model, tmp_path):
    """Anatomy first (CT only), then the planned steps with the vessels held."""
    from ablation2d.model import AblationCNCA
    torch.manual_seed(0)
    model = AblationCNCA(12, 2, 2, anatomy_steps=5).eval()
    for sub in model.sub_models:                     # the untrained model is a no-op
        torch.nn.init.normal_(sub[2].weight, 0.0, 0.03)
    real = np.load(ROOT / "data" / "real_slice.npz")
    anat = from_segmentations(real["seg_liver"], real["seg_vessel"], hounsfield=real["hounsfield"])
    path = tmp_path / "planner.html"
    path.write_text("<html><body>" + BrowserPlanner([("two-phase", model, 0.5, 6)],
                                                    [slice_from_anatomy(anat)]).html() + "</body></html>")
    page = page_and_model[0]                          # reuse the module's browser
    page.goto(path.as_uri())
    page.wait_for_timeout(500)
    _call(page, f"r.nca.pause(); r.nca.setNeedles({NEEDLES});")
    inp = np.array(_call(page, "return r.nca.input();"), np.float32).reshape(128, 128, 4)
    js = _state(page, 4, model.channels)
    x = torch.from_numpy(inp[..., :3].transpose(2, 0, 1).copy())[None]
    with torch.no_grad():
        _, ref = model(x, steps=4, return_state=True)
    assert np.abs(js - ref[0].numpy()).max() < 1e-4
