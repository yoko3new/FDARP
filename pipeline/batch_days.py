"""
Group classification and SRS distances over many days, one frame per day
(default 00:00, where SRS positions need no rotation). Used to check the
association threshold on a larger sample and to find 1:N / N:1 cases.

Example:
    python batch_days.py --start 20141001 --end 20141031
"""

import argparse
from collections import defaultdict

import numpy as np

import paths
from association import CLASSES, associate_frame, classify
from pathlib import Path

from solar_utils import frames_for_day, load_frame, parse_date, parse_srs
from srs_distance import rate_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--end", help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--time", default="0000", help="preferred frame per day")
    ap.add_argument("--fits-root", default=str(paths.FITS))
    ap.add_argument("--h5-dir", default=str(paths.detection_dir("limb")))
    ap.add_argument("--threshold", type=float, default=3.0, help="degrees")
    ap.add_argument("--subsample", type=int, default=4)
    args = ap.parse_args()

    start = f"{parse_date(args.start):%Y%m%d}" if args.start else None
    end = f"{parse_date(args.end):%Y%m%d}" if args.end else None
    days = sorted(p.name for p in Path(args.fits_root).iterdir() if p.is_dir())
    days = [d for d in days if (not start or d >= start) and (not end or d <= end)]

    totals, cases, rows, n_days = defaultdict(int), [], [], 0
    print(f"{'date':>10}{'frame':>7}{'comps':>7}{'regions':>9}"
          + "".join(f"{c:>6}" for c in CLASSES))
    for day in days:
        times = frames_for_day(day, args.fits_root)
        if not times or not paths.srs_file(day).exists():
            continue
        t = args.time if args.time in times else times[0]
        frame = load_frame(day, t, args.fits_root, args.h5_dir)
        if frame is None or frame["mask"] is None:
            continue

        res = associate_frame(frame, parse_srs(paths.srs_file(day)),
                              args.threshold, args.subsample)
        n_days += 1
        for k, v in res["counts"].items():
            totals[k] += v
        cases += [(day, classify(g), g) for g in res["groups"]
                  if classify(g) in ("1:N", "N:1", "M:N")]
        rows += [dict(r, dist=min(res["distances"][r["number"]].values(), default=np.inf))
                 for r in res["regions"]]
        print(f"{day:>10}{t:>7}{len(res['comp_ids']):>7}{len(res['regions']):>9}"
              + "".join(f"{res['counts'].get(c, 0):>6}" for c in CLASSES))

    if n_days == 0:
        raise SystemExit("no day processed; check --fits-root, --h5-dir and SRS files")
    print(f"{'total':>33}" + "".join(f"{totals[c]:>6}" for c in CLASSES))

    print("\nnon-trivial groups")
    for day, cls, g in cases:
        print(f"  {day}  {cls}: components {['C%d' % c for c in g['components']]} "
              f"regions {g['regions']}")
    if not cases:
        print("  none")

    print(f"\nassociation rate vs threshold ({len(rows)} regions)")
    if not rows:
        print("  no visible SRS regions; nothing to measure")
        return
    rate_table(rows)

    print(f"\nsunspot regions beyond {args.threshold:g} deg, by |longitude|")
    for lo, hi in ((0, 30), (30, 60), (60, 90)):
        d = np.array([r["dist"] for r in rows
                      if r["section"] == "I" and lo <= abs(r["lon"]) < hi])
        if len(d):
            print(f"  |lon| {lo:>2}-{hi:<2}  n={len(d):>4}  "
                  f"{100 * np.mean(d > args.threshold):5.1f}%")


if __name__ == "__main__":
    main()
