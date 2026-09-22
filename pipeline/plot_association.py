"""
Show which NOAA SRS regions are associated with detected ARs at two thresholds
(great-circle degrees). Associated regions are filled and linked to the nearest
detected pixel; the others stay hollow with a dotted circle at the threshold.

Examples:
    python plot_association.py --date 20141020 --time 0000
    python plot_association.py --date 20141020 --all --thr-left 2 --thr-right 3
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from association import nearest_on_surface
from solar_utils import (C_DETECT, C_LIMB, C_PLAGE, C_SPOT, advance_longitudes,
                         frames_for_day, latlon_to_pixel, load_frame, parse_srs,
                         require_srs, parse_date)


def draw_panel(ax, frame, regions, nearest, threshold, vlim):
    cx, cy, r_px = frame["cx"], frame["cy"], frame["r_px"]
    ax.imshow(frame["mag"], origin="lower", cmap="gray", vmin=-vlim, vmax=vlim)
    ax.contour(frame["mask"], levels=[0.5], colors=C_DETECT, linewidths=0.9)
    thr_px = np.deg2rad(threshold) * r_px      # circle radius, valid near disk centre

    hit = {"I": 0, "IA": 0}
    tot = {"I": 0, "IA": 0}
    for r, (nx, ny, d) in zip(regions, nearest):
        spot = r["section"] == "I"
        col = C_SPOT if spot else C_PLAGE
        tot[r["section"]] += 1
        if d <= threshold:
            hit[r["section"]] += 1
            ax.plot(r["x"], r["y"], "o", mfc=col, mec=col, ms=14 if spot else 11, alpha=0.85)
            if nx is not None and d > 0.05:
                ax.plot([r["x"], nx], [r["y"], ny], "-", color=col, lw=1.6)
        else:
            ax.plot(r["x"], r["y"], "o", mfc="none", mec=col,
                    ms=16 if spot else 13, mew=2.0 if spot else 1.6)
            ax.add_patch(plt.Circle((r["x"], r["y"]), thr_px, fill=False,
                                    color=col, ls=":", lw=1.0))
        label = r["number"] + (f"\n{r['mag']}" if r["mag"] else "" if spot else "\nplage")
        ax.text(r["x"] + 42, r["y"] + 42, label, color=col,
                fontsize=8.5, fontweight="bold", linespacing=1.1)

    ax.add_patch(plt.Circle((cx, cy), r_px, fill=False, color=C_LIMB, ls="--", lw=0.9))
    ax.set_title(f"threshold {threshold:g} deg\nsunspot regions {hit['I']}/{tot['I']}, "
                 f"plages {hit['IA']}/{tot['IA']} associated", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    return hit, tot


def make_figure(date, time, srs_regions, args):
    frame = load_frame(date, time, args.fits_root, args.h5_dir)
    if frame is None or frame["mask"] is None:
        print(f"skip {date}_{time}: missing input")
        return None

    hours = int(time[:2]) + int(time[2:]) / 60.0
    regions = advance_longitudes(srs_regions, hours)
    xs, ys, vis = latlon_to_pixel([r["lat"] for r in regions],
                                  [r["lon"] for r in regions],
                                  frame["r_px"], frame["cx"], frame["cy"], frame["b0"])
    shown = [dict(r, x=x, y=y) for r, x, y, v in zip(regions, xs, ys, vis) if v]
    nearest = [nearest_on_surface(frame["mask"], r["lat"], r["lon"], frame["r_px"],
                                  frame["cx"], frame["cy"], frame["b0"], args.subsample)
               for r in shown]

    fig, axes = plt.subplots(1, 2, figsize=(22, 11.5))
    res = [draw_panel(ax, frame, shown, nearest, thr, args.vlim)
           for ax, thr in zip(axes, (args.thr_left, args.thr_right))]
    fig.suptitle(f"{frame['stamp']}   filled + line: associated   "
                 f"hollow + dotted circle: not associated", fontsize=12)

    out = Path(args.out_dir) / date / f"{frame['stamp']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--time", default="0000", help="HHMM")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--srs-file")
    ap.add_argument("--fits-root", default=str(paths.FITS))
    ap.add_argument("--h5-dir", default=str(paths.detection_dir("limb")))
    ap.add_argument("--thr-left", type=float, default=2.0)
    ap.add_argument("--thr-right", type=float, default=3.0)
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--out-dir", default=str(paths.FIG_ASSOC))
    ap.add_argument("--vlim", type=float, default=200.0)
    args = ap.parse_args()
    args.date = f"{parse_date(args.date):%Y%m%d}"

    regions = parse_srs(require_srs(args.date, args.srs_file))
    times = frames_for_day(args.date, args.fits_root) if args.all else [args.time]

    total = [[{"I": 0, "IA": 0}, {"I": 0, "IA": 0}] for _ in range(2)]
    n_done = 0
    for t in times:
        res = make_figure(args.date, t, regions, args)
        if res is None:
            continue
        n_done += 1
        for k, (hit, tot) in enumerate(res):
            for s in ("I", "IA"):
                total[k][0][s] += hit[s]
                total[k][1][s] += tot[s]

    if n_done == 0:
        raise SystemExit("no frame processed; check --fits-root and --h5-dir")
    for thr, (hit, tot) in zip((args.thr_left, args.thr_right), total):
        print(f"{thr:g} deg: sunspot regions {hit['I']}/{tot['I']}, "
              f"plages {hit['IA']}/{tot['IA']}")


if __name__ == "__main__":
    main()
