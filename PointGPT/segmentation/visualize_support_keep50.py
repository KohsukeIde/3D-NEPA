#!/usr/bin/env python
"""
ShapeNetPart support-stress figure export (NeurIPS-style).

For each object, under ``random/``, ``semantic/``, ``structured/``:

  - ``before_drop.png`` — full subsampled cloud
  - ``after_drop.png`` — forward cloud after ``stress_one`` (same protocol as eval), single neutral gray
  - ``combined.png`` — same index space as ``before_drop``; **kept = same gray** as above;
    **dropped = rust red** (reserved; do not reuse for semantic parts)

**random / structured:** before and after are uniform gray; combined = gray + drop red.

**semantic:** before = **part-ID colors** (Paul Tol–style qualitative, avoids gray + drop red);
after = uniform gray on forward cloud; top-level ``combined`` = **same kept gray** as random +
drop red (cross-condition consistency).

Additionally under ``semantic/part_<id>/combined.png``: only points of that part ID are drawn;
**kept = that part’s semantic color**, **dropped = rust red** (two colors per figure).

Color rationale (publication / accessibility): avoid rainbow; test grayscale legibility; avoid
red–green-only encoding. See Crameri et al., *Current Protocols* (2024),
https://doi.org/10.1002/cpz1.1126 ; Nature Commun. commentary on colour misuse
https://doi.org/10.1038/s41467-020-19160-7 .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import PartNormalDataset
from eval_shapenetpart_support_stress import build_forward_points, support_indices_one
from pointnet_util import pc_normalize


CONDITION_OFFSETS = {
    "random_keep50": 2,
    "structured_keep50": 6,
    "part_keep50_per_part": 19,
}

# Publication-friendly: one neutral for "geometry / kept", one reserved warm hue for "removed".
# Kept gray chosen for ~white background + grayscale print; drop color not green (CVD).
KEPT_GRAY_RGB = np.asarray((0.35, 0.36, 0.38), dtype=np.float64)  # ~ #5A5C61
DROP_RGB = np.asarray((0.72, 0.18, 0.12), dtype=np.float64)  # rust / vermillion-ish, not pure #FF0000

# Paul Tol–style qualitative (muted subset), ordered; none should match KEPT_GRAY or DROP_RGB closely.
TOL_QUALITATIVE_HEX = (
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#DDCC77",
    "#999933",
    "#AA4499",
    "#882255",
    "#661100",
    "#1177AA",
)

CONDITION_KEY_TO_SPEC = (
    ("random", "random_keep50"),
    ("semantic", "part_keep50_per_part"),
    ("structured", "structured_keep50"),
)


def hide_3d_coordinate_frame(ax, *, proj_type: str = "persp", transparent_bg: bool = False):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    clear = (1.0, 1.0, 1.0, 0.0)
    pane_edge = clear if transparent_bg else "white"
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(pane_edge)
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        if transparent_bg:
            for fn in (
                "set_alpha",
                "set_facecolor",
                "set_edgecolor",
            ):
                if hasattr(axis.pane, fn):
                    try:
                        if fn == "set_alpha":
                            getattr(axis.pane, fn)(0.0)
                        else:
                            getattr(axis.pane, fn)(clear)
                    except Exception:
                        pass
            try:
                axis.pane.set_visible(False)
            except Exception:
                pass
            try:
                axis.set_pane_color(clear)
            except Exception:
                pass
    if transparent_bg:
        try:
            ax.patch.set_alpha(0.0)
            ax.patch.set_facecolor(clear)
            ax.patch.set_visible(False)
        except Exception:
            pass
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis._axinfo["grid"]["color"] = clear  # noqa: SLF001
            except Exception:
                pass
        for attr in ("w_xaxis", "w_yaxis", "w_zaxis", "_w_xaxis", "_w_yaxis", "_w_zaxis"):
            w = getattr(ax, attr, None)
            if w is None:
                continue
            try:
                w.set_alpha(0.0)
                w.set_visible(False)
            except Exception:
                pass
    try:
        ax.set_proj_type(proj_type)
    except Exception:
        pass


# (elev, azim, roll) — matplotlib: elev above xy plane, azim rotation about plot z.
# ``plan_xy``: look onto the XY plane from +Z (bird’s-eye if z is “up” in data).
# ``plan_xz``: look onto XZ from +Y (use if the object reads “side-on” under plan_xy).
VIEW_PRESETS: dict[str, tuple[float, float, float]] = {
    "plan_xy": (90.0, 0.0, 0.0),
    "plan_xz": (0.0, -90.0, 0.0),
    "plan_yz": (0.0, 0.0, 0.0),
}


def dematte_white_png(path: Path) -> None:
    """
    Matplotlib 3D + transparent save often leaves RGB≈255 in semi-transparent edge pixels
    (white matting). Recompute straight-alpha RGB assuming compositing over white (255).
    Only touches pixels that look like **white-ish fringes** (high RGB, non-opaque alpha).
    """
    im = np.array(Image.open(path).convert("RGBA"), dtype=np.float32)
    c = im[..., :3]
    a = im[..., 3:4] / 255.0
    out = im.copy()
    wh = (c[..., 0] + c[..., 1] + c[..., 2]) / 3.0
    m = (im[..., 3] > 2.0) & (im[..., 3] < 254.0) & (wh > 200.0)
    if np.any(m):
        aa = np.maximum(a[m], 1e-6)
        c2 = (c[m] - 255.0 * (1.0 - aa)) / aa
        out[m, :3] = np.clip(c2, 0.0, 255.0)
    out[im[..., 3] < 1.0, :3] = 0.0
    Image.fromarray(np.round(out).astype(np.uint8), "RGBA").save(path)


def auto_plan_view_elev_azim_roll(stacked: np.ndarray) -> tuple[float, float, float]:
    """Pick a view down the **shortest** bbox axis so the widest planar extent fills the figure."""
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = hi - lo
    j = int(np.argmin(span))
    if j == 2:
        return 90.0, 0.0, 0.0
    if j == 1:
        return 0.0, -90.0, 0.0
    return 0.0, 0.0, 0.0


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.asarray([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def part_id_colors(seg: np.ndarray) -> dict[int, np.ndarray]:
    labels = sorted(int(x) for x in np.unique(seg))
    out: dict[int, np.ndarray] = {}
    for i, lab in enumerate(labels):
        hx = TOL_QUALITATIVE_HEX[i % len(TOL_QUALITATIVE_HEX)]
        out[lab] = hex_to_rgb(hx)
    return out


def load_sample_deterministic(
    ds: PartNormalDataset,
    index: int,
    npoints: int,
    sample_rng: np.random.RandomState,
    *,
    eval_like_subsample: bool,
):
    cat, path = ds.datapath[index]
    data = np.loadtxt(path).astype(np.float32)
    point_set = data[:, 0:3].copy()
    seg = data[:, -1].astype(np.int64)
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    n = len(seg)
    if eval_like_subsample or n < npoints:
        choice = sample_rng.choice(n, npoints, replace=True)
    else:
        choice = sample_rng.choice(n, npoints, replace=False)
    return point_set[choice], seg[choice], cat


def aabb_lo_hi(pts_list: list[np.ndarray], *, pad_ratio: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds so every point lies inside limits (with small padding)."""
    stacked = np.concatenate([p for p in pts_list if p.size], axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = float(pad_ratio) * span
    return lo - pad, hi + pad


def lims_from_pts_list(pts_list: list[np.ndarray], *, radius_scale: float = 0.55):
    """Legacy cube from mean + single radius (can crop anisotropic clouds). Prefer aabb_lo_hi."""
    stacked = np.concatenate([p for p in pts_list if p.size], axis=0)
    ctr = stacked.mean(axis=0)
    span = float((stacked.max(axis=0) - stacked.min(axis=0)).max())
    span = max(span, 1e-6)
    r = float(radius_scale) * span
    return ctr, r


def scatter_cloud(
    ax,
    pts: np.ndarray,
    rgb: np.ndarray,
    pt_size: float,
    lo: np.ndarray,
    hi: np.ndarray,
    elev: float,
    azim: float,
    *,
    roll: float = 0.0,
    proj_type: str = "persp",
    transparent_bg: bool = False,
):
    hide_3d_coordinate_frame(ax, proj_type=proj_type, transparent_bg=transparent_bg)
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=pt_size, alpha=0.92, linewidths=0, depthshade=False
    )
    ax.set_xlim3d(float(lo[0]), float(hi[0]))
    ax.set_ylim3d(float(lo[1]), float(hi[1]))
    ax.set_zlim3d(float(lo[2]), float(hi[2]))
    ext = hi - lo
    ext = np.maximum(ext, 1e-9)
    try:
        ax.set_box_aspect(ext / float(ext.max()))
    except Exception:
        pass
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    if transparent_bg:
        hide_3d_coordinate_frame(ax, proj_type=proj_type, transparent_bg=True)


def export_three_pngs_per_condition(
    obj_dir: Path,
    folder: str,
    cond: str,
    pts: np.ndarray,
    seg: np.ndarray,
    *,
    seed: int,
    pt_size: float,
    dpi: int,
    elev: float,
    azim: float,
    roll: float,
    local_noise_sigma: float,
    aabb_pad: float,
    proj_type: str,
    transparent_bg: bool,
):
    rng_cond = np.random.RandomState(seed + CONDITION_OFFSETS[cond])
    keep_idx, forward_idx, _ = support_indices_one(
        pts, seg, cond, rng_cond, local_noise_sigma=local_noise_sigma
    )
    pts_fwd = build_forward_points(pts, forward_idx, cond, rng_cond, local_noise_sigma=local_noise_sigma)
    keep_set = {int(x) for x in keep_idx.tolist()}
    n = pts.shape[0]

    sub = obj_dir / folder
    sub.mkdir(parents=True, exist_ok=True)

    lo, hi = aabb_lo_hi([pts, pts_fwd], pad_ratio=aabb_pad)

    # --- before_drop ---
    if cond == "part_keep50_per_part":
        lut = part_id_colors(seg)
        c_before = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            c_before[i] = lut[int(seg[i])]
    else:
        c_before = np.tile(KEPT_GRAY_RGB, (n, 1))

    # --- after_drop (forward geometry, all same gray) ---
    c_after = np.tile(KEPT_GRAY_RGB, (pts_fwd.shape[0], 1))

    # --- combined (original index space): kept gray, dropped red ---
    c_comb = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        c_comb[i] = KEPT_GRAY_RGB if i in keep_set else DROP_RGB

    fc = "none" if transparent_bg else "white"
    for name, pcloud, cols in (
        ("before_drop", pts, c_before),
        ("after_drop", pts_fwd, c_after),
        ("combined", pts, c_comb),
    ):
        fig = plt.figure(figsize=(4.8, 4.8), facecolor=fc)
        if transparent_bg:
            fig.patch.set_alpha(0.0)
            fig.patch.set_facecolor("none")
        ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor=fc)
        ax.set_facecolor(fc)
        scatter_cloud(
            ax,
            pcloud,
            cols,
            pt_size,
            lo,
            hi,
            elev,
            azim,
            roll=roll,
            proj_type=proj_type,
            transparent_bg=transparent_bg,
        )
        fig.tight_layout(pad=0.05)
        out_path = sub / f"{name}.png"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=transparent_bg,
            facecolor="none" if transparent_bg else None,
        )
        plt.close(fig)
        if transparent_bg:
            dematte_white_png(out_path)
        print(out_path)

    if cond == "part_keep50_per_part":
        lut = part_id_colors(seg)
        for part_id in sorted(int(x) for x in np.unique(seg)):
            idxs = np.flatnonzero(seg == part_id)
            if idxs.size == 0:
                continue
            pts_p = pts[idxs]
            part_rgb = lut[part_id]
            c_part = np.zeros((idxs.size, 3), dtype=np.float64)
            for j in range(idxs.size):
                i = int(idxs[j])
                c_part[j] = part_rgb if i in keep_set else DROP_RGB
            part_dir = sub / f"part_{part_id}"
            part_dir.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(4.8, 4.8), facecolor=fc)
            if transparent_bg:
                fig.patch.set_alpha(0.0)
                fig.patch.set_facecolor("none")
            ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor=fc)
            ax.set_facecolor(fc)
            scatter_cloud(
                ax,
                pts_p,
                c_part,
                pt_size,
                lo,
                hi,
                elev,
                azim,
                roll=roll,
                proj_type=proj_type,
                transparent_bg=transparent_bg,
            )
            fig.tight_layout(pad=0.05)
            out_pp = part_dir / "combined.png"
            fig.savefig(
                out_pp,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=transparent_bg,
                facecolor="none" if transparent_bg else None,
            )
            plt.close(fig)
            if transparent_bg:
                dematte_white_png(out_pp)
            print(out_pp)


def export_hierarchical_object(
    out_root: Path,
    cat: str,
    ds_index: int,
    pts: np.ndarray,
    seg: np.ndarray,
    *,
    seed: int,
    pt_size: float,
    dpi: int,
    view: str,
    elev: float,
    azim: float,
    roll: float,
    local_noise_sigma: float,
    aabb_pad: float,
    proj_type: str,
    transparent_bg: bool,
):
    safe_cat = cat.replace(" ", "_")
    obj_dir = out_root / f"{safe_cat}_idx{ds_index}"
    ev, az, rl = elev, azim, roll
    if view == "auto":
        rng_x = np.random.RandomState(seed + CONDITION_OFFSETS["random_keep50"])
        _ki, fi, _ = support_indices_one(
            pts, seg, "random_keep50", rng_x, local_noise_sigma=local_noise_sigma
        )
        pf = build_forward_points(pts, fi, "random_keep50", rng_x, local_noise_sigma=local_noise_sigma)
        ev, az, rl = auto_plan_view_elev_azim_roll(np.vstack((pts, pf)))
    elif view in VIEW_PRESETS:
        ev, az, rl = VIEW_PRESETS[view]
    for folder, cond in CONDITION_KEY_TO_SPEC:
        export_three_pngs_per_condition(
            obj_dir,
            folder,
            cond,
            pts,
            seg,
            seed=seed,
            pt_size=pt_size,
            dpi=dpi,
            aabb_pad=aabb_pad,
            elev=ev,
            azim=az,
            roll=rl,
            local_noise_sigma=local_noise_sigma,
            proj_type=proj_type,
            transparent_bg=transparent_bg,
        )


def per_part_retention_ratios(seg: np.ndarray, keep_idx: np.ndarray) -> list[float]:
    keep_set = {int(x) for x in keep_idx.tolist()}
    ratios: list[float] = []
    for lab in np.unique(seg):
        idx = np.flatnonzero(seg == lab)
        if idx.size == 0:
            continue
        kept = sum(1 for i in idx if i in keep_set)
        ratios.append(kept / float(idx.size))
    return ratios


def random_part_imbalance_score(seg: np.ndarray, keep_idx_random: np.ndarray) -> float:
    ratios = per_part_retention_ratios(seg, keep_idx_random)
    if len(ratios) < 3:
        return -1.0
    std = float(np.std(ratios))
    worst = float(min(ratios))
    return std + 1.2 * max(0.0, 0.42 - worst)


def discover_contrastive_indices(
    ds: PartNormalDataset,
    *,
    npoint: int,
    sample_seed: int,
    condition_seed: int,
    eval_like_subsample: bool,
    max_scan: int,
    min_points_per_part: int,
) -> list[tuple[float, int, str]]:
    rng_off = CONDITION_OFFSETS["random_keep50"]
    ranked: list[tuple[float, int, str]] = []
    n = min(len(ds), max_scan)
    for ds_index in range(n):
        sample_rng = np.random.RandomState(sample_seed + ds_index)
        try:
            pts, seg, cat = load_sample_deterministic(
                ds, ds_index, npoint, sample_rng, eval_like_subsample=eval_like_subsample
            )
        except Exception:
            continue
        counts = [np.sum(seg == lab) for lab in np.unique(seg)]
        if len(counts) < 3 or min(counts) < min_points_per_part:
            continue
        rng_r = np.random.RandomState(condition_seed + rng_off)
        keep_r, _, _ = support_indices_one(pts, seg, "random_keep50", rng_r, local_noise_sigma=0.08)
        score = random_part_imbalance_score(seg, keep_r)
        if score < 0:
            continue
        ranked.append((score, ds_index, cat))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def resolve_slug_to_index(ds: PartNormalDataset, slug: str) -> tuple[int, str] | None:
    """Match folder basename like ``Airplane_idx0`` to dataset index."""
    for i in range(len(ds)):
        cat = ds.datapath[i][0]
        safe = cat.replace(" ", "_")
        if f"{safe}_idx{i}" == slug:
            return i, cat
    return None


def pick_indices_by_category(ds: PartNormalDataset, max_categories: int):
    seen = set()
    out = []
    for i in range(len(ds)):
        cat = ds.datapath[i][0]
        if cat in seen:
            continue
        seen.add(cat)
        out.append(i)
        if len(out) >= max_categories:
            break
    return out


def plot_flat_three_panel(
    out_path: Path,
    cat: str,
    pts: np.ndarray,
    seg: np.ndarray,
    *,
    seed: int,
    pt_size: float,
    aabb_pad: float = 0.02,
    transparent_bg: bool = False,
):
    """Legacy: three panels after-drop only (combined style)."""
    lo, hi = aabb_lo_hi([pts], pad_ratio=aabb_pad)
    elev, azim, roll = 20.0, -60.0, 0.0
    fc = "none" if transparent_bg else "white"
    fig = plt.figure(figsize=(12.5, 4.0), facecolor=fc)
    if transparent_bg:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
    fig.suptitle(cat, fontsize=11)
    for col, (folder, cond) in enumerate(CONDITION_KEY_TO_SPEC):
        rng = np.random.RandomState(seed + CONDITION_OFFSETS[cond])
        keep_idx, _, _ = support_indices_one(pts, seg, cond, rng, local_noise_sigma=0.08)
        keep_set = {int(x) for x in keep_idx.tolist()}
        n = pts.shape[0]
        c = np.zeros((n, 3), dtype=np.float64)
        if cond == "part_keep50_per_part":
            lut = part_id_colors(seg)
            for i in range(n):
                c[i] = lut[int(seg[i])] if i in keep_set else DROP_RGB
        else:
            for i in range(n):
                c[i] = KEPT_GRAY_RGB if i in keep_set else DROP_RGB
        ax = fig.add_subplot(1, 3, col + 1, projection="3d", facecolor=fc)
        ax.set_facecolor(fc)
        scatter_cloud(
            ax, pts, c, pt_size, lo, hi, elev, azim, roll=roll, proj_type="persp", transparent_bg=transparent_bg
        )
        ax.set_title(f"{folder} (combined)", fontsize=10)
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
        transparent=transparent_bg,
        facecolor="none" if transparent_bg else None,
    )
    plt.close(fig)
    if transparent_bg:
        dematte_white_png(out_path)
    print(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="ShapeNetPart root")
    p.add_argument("--out", required=True, help="Output directory root")
    p.add_argument("--npoint", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--sample_seed", type=int, default=12345)
    p.add_argument("--mode", choices=("default", "contrastive"), default="default")
    p.add_argument("--max_scan", type=int, default=3000)
    p.add_argument("--min_points_per_part", type=int, default=24)
    p.add_argument("--eval_like_subsample", action="store_true")
    p.add_argument("--layout", choices=("hierarchical", "flat"), default="hierarchical")
    p.add_argument("--pt_size", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=300, help="NeurIPS-style raster (e.g. 300–600)")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument(
        "--radius_scale",
        type=float,
        default=0.55,
        help="Used only when --aabb_pad is omitted: smaller => tighter padding around AABB.",
    )
    p.add_argument(
        "--aabb_pad",
        type=float,
        default=None,
        help="Axis-aligned padding as fraction of per-axis span (e.g. 0.02). Default derives from --radius_scale.",
    )
    p.add_argument(
        "--view",
        default="",
        help="Camera preset: plan_xy (down +Z), plan_xz (along ±Y), plan_yz (along ±X), auto (shortest bbox axis), or empty for --elev/--azim.",
    )
    p.add_argument("--proj", default="persp", choices=("persp", "ortho"), help="3D projection type")
    p.add_argument("--roll", type=float, default=0.0, help="Roll angle (degrees); needs recent matplotlib")
    p.add_argument(
        "--transparent_bg",
        action="store_true",
        help="PNG with transparent figure/axes background (for slides / compositing).",
    )
    p.add_argument(
        "--only_slug",
        default="",
        help="If set (e.g. Airplane_idx0), export only that object folder and skip others.",
    )
    p.add_argument("--local_noise_sigma", type=float, default=0.08)
    args = p.parse_args()

    if args.aabb_pad is not None:
        aabb_pad = float(args.aabb_pad)
    else:
        aabb_pad = float(np.clip(0.018 * (0.55 / max(args.radius_scale, 0.12)), 0.004, 0.08))

    valid_views = {"", "auto", *VIEW_PRESETS.keys()}
    if args.view not in valid_views:
        print(f"[error] unknown --view {args.view!r}; choose from {sorted(valid_views - {''})} or leave empty", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PartNormalDataset(root=args.root, npoints=args.npoint, split="val", normal_channel=False)

    if args.mode == "contrastive":
        ranked = discover_contrastive_indices(
            ds,
            npoint=args.npoint,
            sample_seed=args.sample_seed,
            condition_seed=args.seed,
            eval_like_subsample=args.eval_like_subsample,
            max_scan=args.max_scan,
            min_points_per_part=args.min_points_per_part,
        )
        prefer = [
            "Chair", "Table", "Car", "Airplane", "Lamp", "Motorbike", "Guitar", "Skateboard",
            "Knife", "Pistol", "Bag", "Cap", "Earphone", "Laptop", "Rocket",
        ]
        cat_rank = {c: i for i, c in enumerate(prefer)}
        chosen: list[tuple[float, int, str]] = []
        seen_cat: set[str] = set()
        for row in ranked:
            if len(chosen) >= args.num_samples:
                break
            _, _, cat = row
            if cat not in seen_cat:
                chosen.append(row)
                seen_cat.add(cat)
        for row in ranked:
            if len(chosen) >= args.num_samples:
                break
            if row not in chosen:
                chosen.append(row)
        chosen.sort(key=lambda x: (cat_rank.get(x[2], 99), -x[0]))
        chosen = chosen[: args.num_samples]
        work_list = [(idx, cat) for _, idx, cat in chosen]
    else:
        work_list = [(idx, ds.datapath[idx][0]) for idx in pick_indices_by_category(ds, args.num_samples)]

    if args.only_slug:
        hit = resolve_slug_to_index(ds, args.only_slug)
        if hit is None:
            print(f"[error] no dataset index matches slug: {args.only_slug}", file=sys.stderr)
            sys.exit(2)
        work_list = [hit]

    for ds_index, cat in work_list:
        sample_rng = np.random.RandomState(args.sample_seed + ds_index)
        pts, seg, cat_loaded = load_sample_deterministic(
            ds, ds_index, args.npoint, sample_rng, eval_like_subsample=args.eval_like_subsample
        )
        cat = cat_loaded

        if args.layout == "hierarchical":
            export_hierarchical_object(
                out_dir,
                cat,
                ds_index,
                pts,
                seg,
                seed=args.seed,
                pt_size=args.pt_size,
                dpi=args.dpi,
                view=args.view,
                elev=args.elev,
                azim=args.azim,
                roll=args.roll,
                local_noise_sigma=args.local_noise_sigma,
                aabb_pad=aabb_pad,
                proj_type=args.proj,
                transparent_bg=args.transparent_bg,
            )
        else:
            safe_cat = cat.replace(" ", "_")
            plot_flat_three_panel(
                out_dir / f"keep50_{safe_cat}_idx{ds_index}_after_only.png",
                cat,
                pts,
                seg,
                seed=args.seed,
                pt_size=args.pt_size,
                aabb_pad=aabb_pad,
                transparent_bg=args.transparent_bg,
            )


if __name__ == "__main__":
    main()
