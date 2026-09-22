"""
Scan SRS files for days likely to show one component covering several NOAA
regions (close pairs) or one region breaking into several components (large
longitudinal extent), so magnetograms only need to be fetched for those days.

Only section I regions within --max-lon of the central meridian are used.

Example:
    python scan_srs.py --start 20120101 --end 20151231
"""

import argparse
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import paths
from solar_utils import great_circle_deg, parse_date, parse_srs


def day_range(start, end):
    d, d1 = parse_date(start), parse_date(end)
    while d <= d1:
        yield f"{d:%Y%m%d}"
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--pair-deg", type=float, default=8.0,
                    help="report region pairs closer than this (deg)")
    ap.add_argument("--wide-deg", type=int, default=15,
                    help="report regions with LL extent >= this (deg)")
    ap.add_argument("--max-lon", type=float, default=60.0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=str(paths.RESULTS / "srs_candidates.csv"))
    args = ap.parse_args()

    pairs, wide, n_days = [], [], 0
    for date in day_range(args.start, args.end):
        p = paths.srs_file(date)
        if not p.exists():
            continue
        n_days += 1
        regs = [r for r in parse_srs(p)
                if r["section"] == "I" and abs(r["lon"]) <= args.max_lon]
        for i, a in enumerate(regs):
            for b in regs[i + 1:]:
                d = float(great_circle_deg(a["lat"], a["lon"], b["lat"], b["lon"]))
                if d <= args.pair_deg:
                    pairs.append((date, a, b, d))
        wide += [(date, r) for r in regs if (r["ll"] or 0) >= args.wide_deg]

    pairs.sort(key=lambda z: z[3])
    wide.sort(key=lambda z: -z[1]["ll"])
    print(f"scanned {n_days} days")
    if n_days == 0:
        raise SystemExit("no SRS files in the range; run download_srs.py first")

    print(f"\nclose pairs (<= {args.pair_deg:g} deg): {len(pairs)}")
    print(f"{'date':>10}{'A':>7}{'B':>7}{'sep':>7}{'area A':>8}{'area B':>8}")
    for date, a, b, d in pairs[:args.top]:
        print(f"{date:>10}{a['number']:>7}{b['number']:>7}{d:>7.1f}"
              f"{str(a['area']):>8}{str(b['area']):>8}")

    print(f"\nwide regions (LL >= {args.wide_deg} deg): {len(wide)}")
    print(f"{'date':>10}{'region':>8}{'LL':>5}{'area':>7}  mag")
    for date, r in wide[:args.top]:
        print(f"{date:>10}{r['number']:>8}{r['ll']:>5}{str(r['area']):>7}  {r['mag']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("kind,date,region_a,region_b,sep_deg,ll_deg,area_a,area_b\n")
        for date, a, b, d in pairs:
            fh.write(f"pair,{date},{a['number']},{b['number']},{d:.2f},,"
                     f"{a['area']},{b['area']}\n")
        for date, r in wide:
            fh.write(f"wide,{date},{r['number']},,,{r['ll']},{r['area']},\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
