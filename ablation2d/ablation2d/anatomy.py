"""A 2-D axial slice: liver, hepatic vessels and connective tissue.

Two sources: real patient slices (`from_segmentations`), which the corpus is
built on, and a procedural generator (`sample_anatomy`) for tests and
controlled experiments.

* liver is perfused, and the only place a needle tip may end up;
* vessels act as heat sinks, with varying calibre;
* everything else is connective tissue rather than air, since the solver turns
  perfusion and the vessel sink off on non-biological cells.

`physics.py` extrudes the slice along z and solves in 3-D; the label is the
mid-plane, which makes it a deterministic function of the 2-D picture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device import HEAT_SINK_DIAMETER_MM, tissue_properties

# 128 x 128 at 2 mm: a 256 mm field of view, enough for a liver (~200 mm).
NX = NY = 128
DX_MM = 2.0
FOV_MM = NX * DX_MM

LBL_BACKGROUND, LBL_LIVER, LBL_VESSEL = 0, 1, 2


@dataclass
class Anatomy:
    """One 2-D slice. Arrays are (NY, NX), indexed [y, x]."""

    label: np.ndarray           # uint8, LBL_*
    vessel_radius_mm: np.ndarray  # float32, 0 outside vessels
    dx_mm: float = DX_MM
    source: str = "procedural"
    calibres_mm: tuple = ()     # the calibre each vessel was drawn at
    hounsfield: np.ndarray | None = None   # float32 HU, the picture a radiologist has
    #: solver::Material ordinal per cell, on real slices only.
    material_id: np.ndarray | None = None

    def __post_init__(self):
        # label maps built by hand get a synthetic CT from a fixed seed
        if self.hounsfield is None:
            self.hounsfield = synth_hounsfield(self.label,
                                               np.random.default_rng(0))

    @property
    def shape(self) -> tuple[int, int]:
        return self.label.shape

    @property
    def liver_mask(self) -> np.ndarray:
        return self.label == LBL_LIVER

    @property
    def vessel_mask(self) -> np.ndarray:
        return self.label == LBL_VESSEL

    def physical_channels(self) -> dict[str, np.ndarray]:
        """Per-voxel material fields in solver units. On a real slice they are
        read from the solver's table by material id."""
        h, w = self.shape
        if self.material_id is not None:
            return _channels_from_material_ids(self.material_id)

        props = tissue_properties()
        out = {k: np.zeros((h, w), np.float32)
               for k in ("rho", "c", "k", "sigma", "perfusion")}
        mid = np.zeros((h, w), np.uint8)
        for label_name, lbl in (("background", LBL_BACKGROUND),
                                ("liver", LBL_LIVER),
                                ("vessel", LBL_VESSEL)):
            m = self.label == lbl
            if not m.any():
                continue
            p = props[label_name]
            mid[m] = p["material_id"]
            for k in out:
                out[k][m] = p[k]
        out["material_id"] = mid
        return out

    @property
    def vessel_fraction(self) -> float:
        """Vessel area as a fraction of the organ (liver + vessels)."""
        liv = float(self.liver_mask.sum()) + float(self.vessel_mask.sum())
        return float(self.vessel_mask.sum()) / max(liv, 1.0)

    def summary(self) -> dict:
        px_mm2 = self.dx_mm ** 2
        return {
            "source": self.source,
            "liver_area_cm2": round((float(self.liver_mask.sum())
                                     + float(self.vessel_mask.sum())) * px_mm2 / 100.0, 1),
            "vessel_pct_of_liver": round(100.0 * self.vessel_fraction, 1),
            "calibres_mm": [round(c, 1) for c in self.calibres_mm],
            "heat_sink_px": int((self.vessel_radius_mm * 2 >= HEAT_SINK_DIAMETER_MM).sum()),
        }


#: HU mean and spread per tissue, measured on the demo slice (case-1001, axial
#: 154, portal-venous phase). Liver and vessel overlap almost entirely
#: (d' = 0.23): shape, not intensity, is what separates them.
HU_STATS = {
    "background": (40.0, 20.0),    # soft/connective tissue
    "liver": (127.0, 27.0),
    "vessel": (136.0, 48.0),
}

#: HU range mapped to [0, 1] in the corpus.
HU_LO, HU_HI = -1000.0, 1000.0


#: Vessel-over-liver contrast in HU, sampled per case: scans vary, and the demo
#: slice sits at the low end (d' = 0.17 at 2 mm).
VESSEL_CONTRAST_HU = (1.0, 28.0)


def synth_hounsfield(label: np.ndarray, rng: np.random.Generator,
                     correlated: float = 1.2,
                     vessel_contrast_hu: float | None = None) -> np.ndarray:
    """A plausible CT slice for a label map: per-voxel noise plus a spatially
    correlated term, so parenchyma looks mottled as on a real scan."""
    hu = np.zeros(label.shape, np.float32)
    if vessel_contrast_hu is None:
        vessel_contrast_hu = float(rng.uniform(*VESSEL_CONTRAST_HU))
    for name, lbl in (("background", LBL_BACKGROUND), ("liver", LBL_LIVER),
                      ("vessel", LBL_VESSEL)):
        m = label == lbl
        if not m.any():
            continue
        mu, sd = HU_STATS[name]
        if name == "vessel":
            mu = HU_STATS["liver"][0] + vessel_contrast_hu
        hu[m] = mu + rng.normal(0.0, sd * 0.75, int(m.sum())).astype(np.float32)
    if correlated > 0:
        f = _smooth(rng.normal(0.0, 1.0, label.shape).astype(np.float32), correlated)
        f /= max(float(f.std()), 1e-6)
        hu += f * 18.0
    return hu


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except ImportError:                       # a tiny separable box blur
        k = max(1, int(round(sigma)))
        out = a.copy()
        for _ in range(2):
            out = np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), "same"),
                0, out)
            out = np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), "same"),
                1, out)
        return out


def normalise_hu(hu: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(hu, np.float32) - HU_LO) / (HU_HI - HU_LO),
                   0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# procedural generation
# --------------------------------------------------------------------------- #
def _smooth_blob(rng, cx, cy, r_mean_px, n_harm=6, wobble=0.22, shape=(NY, NX)):
    """A closed star-shaped region: a circle whose radius is a smooth random
    function of angle. Six harmonics is enough to look organ-like and few enough
    that the boundary never self-intersects."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(yy - cy, xx - cx)
    amp = rng.normal(0.0, 1.0, n_harm) * wobble / np.arange(1, n_harm + 1)
    pha = rng.uniform(0, 2 * np.pi, n_harm)
    r_theta = np.full_like(ang, float(r_mean_px))
    for i in range(n_harm):
        r_theta *= 1.0 + amp[i] * np.cos((i + 1) * ang + pha[i])
    return rad < r_theta


def _tube(points_px, radius_px, shape=(NY, NX)):
    """Distance-to-polyline <= radius. Returns (mask, distance_px).

    Painting the tube as a distance field rather than stamping discs is what
    makes the radius field exact: for a tube, the inscribed radius at a voxel is
    the tube radius, which is the quantity the sink term wants.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.inf, np.float32)
    pts = np.asarray(points_px, np.float32)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))
        np.minimum(best, d, out=best)
    return best <= radius_px, best


def _wander(rng, start, n_steps, step_px, turn_sigma, shape=(NY, NX)):
    """A gently curving centreline. Vessels bend; they do not zig-zag."""
    h, w = shape
    p = np.asarray(start, np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    pts = [p.copy()]
    for _ in range(n_steps):
        theta += rng.normal(0.0, turn_sigma)
        p = p + step_px * np.array([np.cos(theta), np.sin(theta)], np.float32)
        p[0] = np.clip(p[0], 1, w - 2)
        p[1] = np.clip(p[1], 1, h - 2)
        pts.append(p.copy())
    return np.stack(pts)


def sample_anatomy(rng: np.random.Generator, *, nx: int = NX, ny: int = NY,
                   dx_mm: float = DX_MM) -> Anatomy:
    """One random liver slice.

    Liver area 90-250 cm^2; 2-6 vessels, at most one trunk of 8-20 mm and
    branches of 2.5-7 mm on both sides of the 3 mm heat-sink threshold; vessels
    3-20 % of the organ (the demo slice has 13.4 %), redrawn otherwise. Ranges
    were set so the real demo slice falls inside them.
    """
    shape = (ny, nx)
    px = dx_mm

    # -- liver ---------------------------------------------------------------
    for _ in range(40):
        r_mean_px = rng.uniform(58.0, 95.0) / px      # 58-95 mm mean radius
        cx = rng.uniform(0.36, 0.64) * nx
        cy = rng.uniform(0.36, 0.64) * ny
        liver = _smooth_blob(rng, cx, cy, r_mean_px, shape=shape)
        # Keep the organ off the frame edge: a liver flush against the boundary
        # puts the adiabatic wall inside the ablation zone.
        liver[:2] = liver[-2:] = False
        liver[:, :2] = liver[:, -2:] = False
        area_cm2 = liver.sum() * px * px / 100.0
        if 90.0 <= area_cm2 <= 250.0:
            break

    liver_px = int(liver.sum())
    ys, xs = np.nonzero(liver)

    # -- vessels -------------------------------------------------------------
    for _attempt in range(12):
        label = np.full(shape, LBL_BACKGROUND, np.uint8)
        label[liver] = LBL_LIVER
        radius = np.zeros(shape, np.float32)
        calibres: list[float] = []
        n_ves = int(rng.integers(2, 7))
        has_trunk = rng.random() < 0.75
        for v in range(n_ves):
            j = rng.integers(len(xs))
            start = (float(xs[j]), float(ys[j]))
            trunk = has_trunk and v == 0
            calibre_mm = float(rng.uniform(8.0, 20.0) if trunk
                               else rng.uniform(2.5, 7.0))
            line = _wander(rng, start,
                           n_steps=int(rng.integers(6, 13) if trunk
                                       else rng.integers(4, 10)),
                           step_px=rng.uniform(6.0, 12.0),
                           turn_sigma=rng.uniform(0.25, 0.7), shape=shape)
            tube, _ = _tube(line, 0.5 * calibre_mm / px, shape=shape)
            # A vessel exists only inside the organ. Clipping to the liver also
            # keeps the sink off the background, where it would be a fiction.
            tube &= liver
            if not tube.any():
                continue
            label[tube] = LBL_VESSEL
            # Overlapping branches: the larger calibre wins, which is what an
            # inscribed-radius measurement of the union would return.
            radius[tube] = np.maximum(radius[tube], 0.5 * calibre_mm)
            calibres.append(calibre_mm)
        frac = float((label == LBL_VESSEL).sum()) / max(liver_px, 1)
        if 0.03 <= frac <= 0.20:
            break

    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source="procedural", calibres_mm=tuple(sorted(calibres)),
                   hounsfield=synth_hounsfield(label, rng))


def _material_table(ids) -> dict[int, dict]:
    """`{material_id: properties}` straight from the solver, cached per id set."""
    from .device import material_properties_by_id
    return {int(i): material_properties_by_id(int(i)) for i in ids}


def _channels_from_material_ids(mid: np.ndarray) -> dict[str, np.ndarray]:
    mid = np.asarray(mid, np.uint8)
    out = {k: np.zeros(mid.shape, np.float32)
           for k in ("rho", "c", "k", "sigma", "perfusion")}
    for i, p in _material_table(np.unique(mid)).items():
        m = mid == i
        for k in out:
            out[k][m] = p[k]
    out["material_id"] = mid
    return out


def from_patient_slice(hounsfield, material_id, seg_liver, seg_vessel,
                       dx_mm: float = DX_MM, source: str = "patient") -> "Anatomy":
    """One real axial slice: real CT, real segmentation, real material map."""
    sv = np.asarray(seg_vessel, bool)
    sl = np.asarray(seg_liver, bool) | sv
    label = np.full(sl.shape, LBL_BACKGROUND, np.uint8)
    label[sl] = LBL_LIVER
    label[sv] = LBL_VESSEL
    radius = inscribed_radius_mm(sv, dx_mm)
    cal = 2 * radius[sv]
    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source=source,
                   hounsfield=np.asarray(hounsfield, np.float32),
                   material_id=np.asarray(material_id, np.uint8),
                   calibres_mm=tuple(np.round(np.percentile(
                       cal, [0, 50, 100]), 1)) if cal.size else ())


# --------------------------------------------------------------------------- #
# a real slice, for the demo
# --------------------------------------------------------------------------- #
def from_segmentations(seg_liver: np.ndarray, seg_vessel: np.ndarray,
                       dx_mm: float = DX_MM, source: str = "patient",
                       hounsfield: np.ndarray | None = None,
                       rng: np.random.Generator | None = None) -> Anatomy:
    """An Anatomy from a liver mask and a vessel mask on the workshop grid.

    The organ is their union: in the 1 mm source corpus the masks are disjoint,
    in `data_real/` the liver mask already contains the vessels, and the union
    is right for both. Vessel radii are the inscribed radii of the vessel mask.
    """
    seg_liver = np.asarray(seg_liver, bool)
    seg_vessel = np.asarray(seg_vessel, bool)
    organ = seg_liver | seg_vessel
    label = np.full(organ.shape, LBL_BACKGROUND, np.uint8)
    label[organ] = LBL_LIVER
    label[seg_vessel] = LBL_VESSEL
    radius = inscribed_radius_mm(seg_vessel, dx_mm)
    cal = 2 * radius[seg_vessel]
    hu = (np.asarray(hounsfield, np.float32) if hounsfield is not None
          else synth_hounsfield(label, rng or np.random.default_rng(0)))
    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source=source, hounsfield=hu,
                   calibres_mm=tuple(np.round(np.percentile(
                       cal, [0, 50, 100]), 1)) if cal.size else ())


def inscribed_radius_mm(mask: np.ndarray, dx_mm: float = DX_MM) -> np.ndarray:
    """Distance in mm to the nearest cell outside the mask (the inscribed radius).
    Uses scipy if available, a brute-force fallback otherwise."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(mask)
    except ImportError:
        pad = np.pad(mask, 1, constant_values=False)
        by, bx = np.nonzero(pad[1:-1, 1:-1] & ~(
            pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]))
        yy, xx = np.nonzero(mask)
        d = np.zeros(mask.shape, np.float32)
        if len(by):
            dd = np.hypot(yy[:, None] - by[None, :], xx[:, None] - bx[None, :])
            d[yy, xx] = dd.min(axis=1) + 0.5
    return (d * dx_mm).astype(np.float32)
