"""
Download NOAA SRS (Solar Region Summary) files from the NOAA NCEI archive.

Output: data/srs/<YYYYMMDD>SRS.txt

Examples:
    python download_srs.py --date 20141020
    python download_srs.py --start 2014-10-01 --end 2014-10-31
"""

import argparse
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path

import paths
from solar_utils import parse_date


def fetch(date, out_dir, overwrite=False):
    dest = Path(out_dir) / f"{date}SRS.txt"
    if dest.exists() and not overwrite:
        return "skip"
    try:
        with urllib.request.urlopen(paths.srs_url(date), timeout=30) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  {date}: {e}")
        return "fail"
    if b"Nmbr" not in body:
        print(f"  {date}: not an SRS file")
        return "fail"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--start", help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--end", help="YYYYMMDD or YYYY-MM-DD, inclusive")
    ap.add_argument("--out-dir", default=str(paths.SRS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.date:
        dates = [f"{parse_date(args.date):%Y%m%d}"]
    elif args.start and args.end:
        d, d1 = parse_date(args.start), parse_date(args.end)
        dates = []
        while d <= d1:
            dates.append(f"{d:%Y%m%d}")
            d += timedelta(days=1)
    else:
        raise SystemExit("give --date, or --start and --end")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for d in dates:
        counts[fetch(d, args.out_dir, args.overwrite)] += 1
    print(f"{counts['ok']} downloaded, {counts['skip']} already present, "
          f"{counts['fail']} failed")
    if counts["ok"] + counts["skip"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
