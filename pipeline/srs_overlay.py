"""
Overlay NOAA SRS regions on detected ARs. With --right-dir, a second panel
shows another detection run (e.g. a different limb cut) side by side.

Examples:
    python srs_overlay.py --date 20141020 --time 0000
    python srs_overlay.py --date 20141020 --all \
        --right-dir ../results/detections/angle80 --right-label "80 deg" --right-angle 80
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from solar_utils import (C_CUT, C_DETECT, C_LIMB, C_PLAGE, C_SPOT,
                         advance_longitudes, frames_for_day, latlon_to_pixel,
                         load_frame, parse_srs, require_srs, parse_date)


def load_mask(h5_dir, stamp):
    import h5py
    p = Path(h5_dir) / stamp[:8] / f"{stamp}.magnetogram.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return np.array(f["union_with_intersect"]) > 0


def draw_panel(ax, frame, mask, regions, title, cut_r, vlim):
    cx, cy, r_px = frame["cx"], frame["cy"], frame["r_px"]
    ax.imshow(frame["mag"], origin="lower", cmap="gray", vmin=-vlim, vmax=vlim)
    ax.contour(mask, levels=[0.5], colors=C_DETECT, linewidths=0.9)

    for r in regions:
        spot = r["section"] == "I"
        col = C_SPOT if spot else C_PLAGE
        ax.plot(r["x"], r["y"], "o", mfc="none", mec=col,
                ms=17 if spot else 13, mew=2.0 if spot else 1.6)
        label = r["number"] + (f"\n{r['mag']}" if r["mag"] else "" if spot else "\nplage")
        ax.text(r["x"] + 42, r["y"] + 42, label, color=col,
                fontsize=8.5, fontweight="bold", linespacing=1.1)

    ax.add_patch(plt.Circle((cx, cy), r_px, fill=False, color=C_LIMB, ls="--", lw=0.9))
    if cut_r is not None and cut_r > 0 and abs(cut_r - r_px) > 2:
        ax.add_patch(plt.Circle((cx, cy), cut_r, fill=False, color=C_CUT, lw=1.3))
        title += f"\nlimb {r_px:.0f} px, cut {cut_r:.0f} px"
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])


def cut_radius(r_px, frac=None, angle=None):
    if angle:
        return r_px * np.sin(np.deg2rad(angle))
    if frac:
        return frac * r_px
    return None


def make_figure(date, time, srs_regions, args):
    frame = load_frame(date, time, args.fits_root)
    if frame is None:
        print(f"skip {date}_{time}: no magnetogram")
        return False
    left = load_mask(args.left_dir, frame["stamp"])
    right = load_mask(args.right_dir, frame["stamp"]) if args.right_dir else None
    if left is None or (args.right_dir and right is None):
        print(f"skip {frame['stamp']}: no detection")
        return False

    hours = int(time[:2]) + int(time[2:]) / 60.0
    regions = advance_longitudes(srs_regions, hours)
    xs, ys, vis = latlon_to_pixel([r["lat"] for r in regions],
                                  [r["lon"] for r in regions],
                                  frame["r_px"], frame["cx"], frame["cy"], frame["b0"])
    shown = [dict(r, x=x, y=y) for r, x, y, v in zip(regions, xs, ys, vis) if v]

    n = 2 if right is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 11.5), squeeze=False)
    draw_panel(axes[0, 0], frame, left, shown, args.left_label,
               cut_radius(frame["r_px"], args.left_frac, args.left_angle), args.vlim)
    if right is not None:
        draw_panel(axes[0, 1], frame, right, shown, args.right_label,
                   cut_radius(frame["r_px"], args.right_frac, args.right_angle), args.vlim)

    fig.suptitle(f"{frame['stamp']}   green: detection   red: NOAA sunspot region   "
                 f"orange: NOAA plage\ndisk ({frame['cx']:.1f}, {frame['cy']:.1f}) "
                 f"R={frame['r_px']:.1f} px   B0={frame['b0']:.2f} deg", fontsize=12)
    out = Path(args.out_dir) / date / f"{frame['stamp']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--time", default="0000", help="HHMM")
    ap.add_argument("--all", action="store_true", help="all frames of the day")
    ap.add_argument("--srs-file")
    ap.add_argument("--fits-root", default=str(paths.FITS))
    ap.add_argument("--left-dir", default=str(paths.detection_dir("limb")))
    ap.add_argument("--left-label", default="cut at the limb")
    ap.add_argument("--left-frac", type=float, default=1.0)
    ap.add_argument("--left-angle", type=float)
    ap.add_argument("--right-dir", help="second detection run to compare")
    ap.add_argument("--right-label", default="")
    ap.add_argument("--right-frac", type=float)
    ap.add_argument("--right-angle", type=float)
    ap.add_argument("--out-dir", default=str(paths.FIG_OVERLAY))
    ap.add_argument("--vlim", type=float, default=200.0)
    args = ap.parse_args()
    args.date = f"{parse_date(args.date):%Y%m%d}"

    regions = parse_srs(require_srs(args.date, args.srs_file))
    times = frames_for_day(args.date, args.fits_root) if args.all else [args.time]
    n_done = sum(bool(make_figure(args.date, t, regions, args)) for t in times)
    if n_done == 0:
        raise SystemExit("no figure made; check --fits-root and the detection dirs")


if __name__ == "__main__":
    main()
