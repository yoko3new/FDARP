"""
Classify component / NOAA-region groups (1:1, 1:N, N:1, M:N, 1:0, 0:1) for the
frames of a day, and optionally plot one frame with each component labelled.

Example:
    python check_groups.py --date 20141020 --time 0000 --plot
    python check_groups.py --date 20141020 --all
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paths
from association import CLASSES, associate_frame, classify
from solar_utils import (C_DETECT, C_LIMB, C_PLAGE, C_SPOT, frames_for_day,
                         load_frame, parse_srs, require_srs, parse_date)

CLASS_COLOUR = {"1:1": "white", "1:0": "yellow", "1:N": "magenta",
                "N:1": "cyan", "M:N": "magenta"}
STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]


def plot_groups(frame, res, args):
    fig, ax = plt.subplots(figsize=(13.5, 13.5))
    ax.imshow(frame["mag"], origin="lower", cmap="gray", vmin=-args.vlim, vmax=args.vlim)
    ax.contour(frame["mask"], levels=[0.5], colors=C_DETECT, linewidths=0.9)

    stats, used = res["stats"], set()
    for g in res["groups"]:
        cls = classify(g)
        for c in g["components"]:
            used.add(cls)
            x = stats[c, cv2.CC_STAT_LEFT] + stats[c, cv2.CC_STAT_WIDTH] / 2
            y = stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] / 2
            ax.text(x, y, f"C{c}\n{cls}", color=CLASS_COLOUR[cls], fontsize=8.5,
                    ha="center", va="center", fontweight="bold", path_effects=STROKE)

    for r in res["regions"]:
        col = C_SPOT if r["section"] == "I" else C_PLAGE
        ax.plot(r["x"], r["y"], "o", mfc="none", mec=col, ms=15, mew=1.9)
        ax.text(r["x"] + 42, r["y"] + 42, r["number"], color=col, fontsize=9.5,
                fontweight="bold", path_effects=STROKE)
    ax.add_patch(plt.Circle((frame["cx"], frame["cy"]), frame["r_px"],
                            fill=False, color=C_LIMB, ls="--", lw=0.9))

    handles = [Line2D([0], [0], color=C_DETECT, lw=2, label="detected AR"),
               Line2D([0], [0], marker="o", ls="", mfc="none", mec=C_SPOT, ms=10,
                      label="NOAA region with sunspots"),
               Line2D([0], [0], marker="o", ls="", mfc="none", mec=C_PLAGE, ms=8,
                      label="NOAA plage"),
               Line2D([0], [0], color=C_LIMB, ls="--", label="solar limb")]
    handles += [Line2D([0], [0], marker="s", ls="", mfc=CLASS_COLOUR[c], mec="black",
                       ms=9, label=c) for c in CLASSES if c in used]
    ax.legend(handles=handles, loc="lower right", fontsize=9, facecolor="0.15",
              labelcolor="white", framealpha=0.85)
    ax.set_title(f"{frame['stamp']}   components C<id> and group class   "
                 f"threshold {args.threshold:g} deg", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    out = Path(args.out_dir) / frame["date"] / f"{frame['stamp']}_groups.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--time", default="0000", help="HHMM")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--srs-file")
    ap.add_argument("--fits-root", default=str(paths.FITS))
    ap.add_argument("--h5-dir", default=str(paths.detection_dir("limb")))
    ap.add_argument("--threshold", type=float, default=3.0, help="degrees")
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--plot", action="store_true", help="plot the first frame")
    ap.add_argument("--out-dir", default=str(paths.FIG_GROUPS))
    ap.add_argument("--vlim", type=float, default=200.0)
    args = ap.parse_args()
    args.date = f"{parse_date(args.date):%Y%m%d}"

    srs = parse_srs(require_srs(args.date, args.srs_file))
    times = frames_for_day(args.date, args.fits_root) if args.all else [args.time]

    print(f"{'frame':>15}{'comps':>7}{'regions':>9}" + "".join(f"{c:>6}" for c in CLASSES))
    n_done = 0
    for i, t in enumerate(times):
        frame = load_frame(args.date, t, args.fits_root, args.h5_dir)
        if frame is None or frame["mask"] is None:
            continue
        res = associate_frame(frame, srs, args.threshold, args.subsample)
        n_done += 1
        print(f"{frame['stamp']:>15}{len(res['comp_ids']):>7}{len(res['regions']):>9}"
              + "".join(f"{res['counts'].get(c, 0):>6}" for c in CLASSES))
        for g in res["groups"]:
            if classify(g) in ("1:N", "N:1", "M:N"):
                print(f"{'':>15}  {classify(g)}: components "
                      f"{['C%d' % c for c in g['components']]} regions {g['regions']}")
        if args.plot and n_done == 1:
            plot_groups(frame, res, args)
    if n_done == 0:
        raise SystemExit("no frame processed; check --fits-root and --h5-dir")


if __name__ == "__main__":
    main()
