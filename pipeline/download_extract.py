"""
Download Surya-bench frames and keep only the LoS magnetogram as FITS.

For each timestamp: download the .nc from the public S3 bucket, extract the
hmi_m channel into a compressed FITS the detection code can read, delete the
.nc. Requires the AWS CLI (no credentials needed).

Output: data/fits/<YYYYMMDD>/<YYYYMMDD_HHMM>.magnetogram.fits

Examples:
    python download_extract.py --start 20141020 --end 20141020 --hours 1
    python download_extract.py --start 2014-10-01 --end 2014-10-31 --daily-at 0000
"""

import argparse
import re
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np

import paths
from solar_utils import parse_date

BUCKET = "nasa-surya-bench"
CHANNEL = "hmi_m"

# The netCDF files carry no FITS header. These placeholders only let the C++
# header reader run; the detection measures the disk geometry from the data.
PLACEHOLDER_HEADER = {"CRPIX1": 2048.5, "CRPIX2": 2048.5,
                      "CDELT1": 0.6, "CDELT2": 0.6, "RSUN_OBS": 976.0}


def extract_to_fits(nc_path, fits_path):
    """Write CHANNEL as HDU 1 (compressed image) of a new FITS file."""
    import h5netcdf
    from astropy.io import fits

    with h5netcdf.File(nc_path, "r") as f:
        if CHANNEL not in f.variables:
            print(f"  warning: {CHANNEL} not in {nc_path.name}")
            return False
        data = np.asarray(f.variables[CHANNEL][:], dtype=np.float32)

    hdr = fits.Header()
    for k, v in PLACEHOLDER_HEADER.items():
        hdr[k] = v
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.CompImageHDU(data=data, header=hdr, compression_type="RICE_1")]
                 ).writeto(fits_path, overwrite=True)
    return True


def timestamps(start, end, hours, daily_at):
    """All timestamps from `start` 00:00 through the whole of `end`."""
    t0, stop = parse_date(start), parse_date(end) + timedelta(days=1)
    if t0 >= stop:
        raise SystemExit("--end is before --start")
    if daily_at:
        if not re.fullmatch(r"([01]\d|2[0-3])[0-5]\d", daily_at):
            raise SystemExit(f"bad --daily-at '{daily_at}': use HHMM")
        hh, mm = int(daily_at[:2]), int(daily_at[2:])
        out, d = [], t0
        while d < stop:
            out.append(d.replace(hour=hh, minute=mm))
            d += timedelta(days=1)
        return out
    if hours <= 0:
        raise SystemExit("--hours must be positive")
    out, t = [], t0
    while t < stop:
        out.append(t)
        t += timedelta(hours=hours)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYYMMDD or YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYYMMDD or YYYY-MM-DD, whole day included")
    ap.add_argument("--hours", type=int, default=1, help="cadence in hours")
    ap.add_argument("--daily-at", metavar="HHMM",
                    help="one frame per day at this time (overrides --hours)")
    ap.add_argument("--fits-dir", default=str(paths.FITS))
    ap.add_argument("--tmp-dir", default=str(paths.ROOT / "_nc_tmp"))
    ap.add_argument("--keep-nc", action="store_true")
    args = ap.parse_args()

    fits_dir, tmp_dir = Path(args.fits_dir), Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    done = skip = fail = 0
    for t in timestamps(args.start, args.end, args.hours, args.daily_at):
        stamp = f"{t:%Y%m%d_%H%M}"
        out = fits_dir / f"{t:%Y%m%d}" / f"{stamp}.magnetogram.fits"
        if out.exists():
            skip += 1
            continue

        nc = tmp_dir / f"{stamp}.nc"
        key = f"{t.year}/{t.month:02d}/{stamp}.nc"
        print(f"[{stamp}] downloading")
        r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(nc),
                            "--no-sign-request", "--only-show-errors"])
        if r.returncode != 0 or not nc.exists():
            print("  not available in the bucket")
            fail += 1
            continue

        ok = extract_to_fits(nc, out)
        if not args.keep_nc:
            nc.unlink(missing_ok=True)
        done += ok
        fail += not ok

    if not args.keep_nc:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    print(f"done: {done} written, {skip} already present, {fail} failed")
    # single missing timestamps are normal (quality gaps in Surya-bench)
    if done + skip == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
