"""
Distance (great-circle degrees) from each NOAA SRS region to the nearest
detected AR pixel, over all frames of a day, with association rates for a
range of thresholds.

Example:
    python srs_distance.py --date 20141020 --plot
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from association import nearest_on_surface
from solar_utils import (advance_longitudes, frames_for_day, is_visible,
                         load_frame, parse_srs, require_srs, parse_date)

THRESHOLDS = (0.5, 1, 2, 3, 4, 5, 7, 10, 15)


def frame_distances(date, time, srs_regions, args):
    frame = load_frame(date, time, args.fits_root, args.h5_dir)
    if frame is None or frame["mask"] is None:
        return []
    hours = int(time[:2]) + int(time[2:]) / 60.0
    rows = []
    for r in advance_longitudes(srs_regions, hours):
        if not is_visible(r["lat"], r["lon"], frame["b0"]):
            continue
        _, _, d = nearest_on_surface(frame["mask"], r["lat"], r["lon"], frame["r_px"],
                                     frame["cx"], frame["cy"], frame["b0"], args.subsample)
        rows.append(dict(r, time=time, dist=d))
    return rows


def rate_table(rows, thresholds=THRESHOLDS):
    print(f"{'threshold':>10}{'section I':>12}{'section IA':>12}{'all':>8}")
    for thr in thresholds:
        line = f"{thr:>10g}"
        for sec in ("I", "IA"):
            d = [r["dist"] for r in rows if r["section"] == sec]
            line += f"{100 * np.mean(np.array(d) <= thr):>11.0f}%" if d else f"{'-':>12}"
        line += f"{100 * np.mean(np.array([r['dist'] for r in rows]) <= thr):>7.0f}%"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--srs-file")
    ap.add_argument("--fits-root", default=str(paths.FITS))
    ap.add_argument("--h5-dir", default=str(paths.detection_dir("limb")))
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=str(paths.FIG_DIAG / "srs_distance.png"))
    args = ap.parse_args()
    args.date = f"{parse_date(args.date):%Y%m%d}"

    srs = parse_srs(require_srs(args.date, args.srs_file))
    rows = []
    for t in frames_for_day(args.date, args.fits_root):
        rows += frame_distances(args.date, t, srs, args)
    if not rows:
        raise SystemExit("no measurements; check the paths")

    print(f"{'region':>7}{'sec':>5}{'mean':>8}{'min':>7}{'max':>8}{'frames':>8}")
    for n in sorted({r["number"] for r in rows}):
        d = np.array([r["dist"] for r in rows if r["number"] == n])
        sec = next(r["section"] for r in rows if r["number"] == n)
        print(f"{n:>7}{sec:>5}{d.mean():8.2f}{d.min():7.2f}{d.max():8.2f}{len(d):>8}")
    print()
    rate_table(rows)

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for n in sorted({r["number"] for r in rows}):
            rr = [r for r in rows if r["number"] == n]
            spot = rr[0]["section"] == "I"
            axes[0].plot([int(r["time"][:2]) + int(r["time"][2:]) / 60 for r in rr],
                         [r["dist"] for r in rr], "o-" if spot else "s--",
                         color="tab:red" if spot else "tab:orange", ms=4, lw=1, label=n)
        axes[0].set(xlabel="hour (UT)", ylabel="distance (deg)")
        axes[0].legend(fontsize=8, ncol=2)
        for sec, col in (("I", "tab:red"), ("IA", "tab:orange")):
            d = [r["dist"] for r in rows if r["section"] == sec and np.isfinite(r["dist"])]
            if d:
                axes[1].hist(d, bins=25, alpha=0.65, color=col, label=f"section {sec}")
        axes[1].set(xlabel="distance (deg)", ylabel="count")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=95)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
