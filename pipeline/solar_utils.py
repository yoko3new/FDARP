"""
Shared helpers: SRS parsing, disk geometry, coordinate transforms and frame I/O.

Coordinates are heliographic Stonyhurst (degrees, west positive). Separations
are great-circle distances on the solar surface, not pixel distances: 100 px is
~3.5 deg at disk centre but ~8 deg at W60 because of foreshortening.
"""

import re
from pathlib import Path

import numpy as np

import paths

# Plot colours shared by the figure scripts
C_DETECT = "lime"      # detected AR contours
C_SPOT = "red"         # NOAA section I (regions with sunspots)
C_PLAGE = "orange"     # NOAA section IA (plages without spots)
C_LIMB = "yellow"      # solar limb
C_CUT = "white"        # limb-cut radius

# Synodic rotation rate used to advance SRS longitudes within a day (deg/day)
ROTATION_DEG_PER_DAY = 13.2

_LOCATION_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")


def parse_date(s):
    """Accept YYYYMMDD or YYYY-MM-DD; return a datetime at 00:00."""
    from datetime import datetime
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise SystemExit(f"bad date '{s}': use YYYYMMDD or YYYY-MM-DD")


# ----------------------------------------------------------------- SRS parsing

def parse_location(loc):
    """'S13E43' -> (-13.0, -43.0)."""
    m = _LOCATION_RE.match(loc.strip())
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lat = float(lat) * (-1.0 if ns == "S" else 1.0)
    lon = float(lon) * (-1.0 if ew == "E" else 1.0)
    return lat, lon


def parse_srs(path):
    """
    Parse sections I (regions with sunspots) and IA (plages) of an SRS file.
    Section II lists regions still behind the east limb and is skipped.

    Returns dicts with number, lat, lon, section, area, ll, mag. Section I rows:
        Nmbr Location Lo Area Z LL NN MagType
        2192 S13E43   251 1560 Fkc 22 32 Beta-Gamma-Delta
    Positions are valid at 00:00 UT of the report date.
    """
    regions = []
    section = None
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("I.") and "Sunspot" in s:
                section = "I"
                continue
            if s.startswith("IA."):
                section = "IA"
                continue
            if s.startswith("II."):
                section = "II"
                continue
            if section in (None, "II"):
                continue

            parts = s.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            loc = parse_location(parts[1])
            if loc is None:
                continue

            r = {"number": parts[0], "lat": loc[0], "lon": loc[1],
                 "section": section, "area": None, "ll": None, "mag": None}
            if section == "I":
                if len(parts) >= 4 and parts[3].isdigit():
                    r["area"] = int(parts[3])
                if len(parts) >= 6 and parts[5].isdigit():
                    r["ll"] = int(parts[5])
                if len(parts) >= 8:
                    r["mag"] = " ".join(parts[7:])
            regions.append(r)
    return regions


def require_srs(date, srs_file=None):
    """Return the SRS path for a date, or exit with a download hint."""
    p = Path(srs_file) if srs_file else paths.srs_file(date)
    if not p.exists():
        raise SystemExit(f"SRS file not found: {p}\n"
                         f"Download it with: python download_srs.py --date {date}")
    return p


# --------------------------------------------------------------- disk geometry

def _extent_geometry(xs, ys):
    return ((xs.max() + xs.min()) / 2.0,
            (ys.max() + ys.min()) / 2.0,
            ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4.0)


def detect_disk_scan(mag, threshold=1.0):
    """Disk centre/radius from the non-zero extent of the central row and column."""
    h, w = mag.shape
    xs = np.where(np.abs(mag[h // 2, :]) > threshold)[0]
    ys = np.where(np.abs(mag[:, w // 2]) > threshold)[0]
    if len(xs) == 0 or len(ys) == 0:
        return None
    return _extent_geometry(xs, ys)


def detect_disk_bbox(mag, threshold=1.0):
    """Disk centre/radius from the bounding box of all on-disk pixels."""
    ys, xs = np.nonzero(np.abs(mag) > threshold)
    if len(xs) == 0:
        return None
    return _extent_geometry(xs, ys)


def measure_disk(mag, threshold=1.0, tol=5.0, verbose=False, tag=""):
    """
    Measure the disk with both estimators. Returns (cx, cy, r, flagged), where
    flagged means the estimators disagree by more than `tol` pixels. The scan
    estimate is used. The Surya disk centre is ~27 px off the image centre, so
    it is measured rather than assumed.
    """
    scan = detect_disk_scan(mag, threshold)
    bbox = detect_disk_bbox(mag, threshold)
    if scan is None and bbox is None:
        raise RuntimeError(f"{tag}: could not locate the solar disk")
    if scan is None:
        return (*bbox, True)
    if bbox is None:
        return (*scan, True)

    diff = max(abs(a - b) for a, b in zip(scan, bbox))
    flagged = diff > tol
    if verbose or flagged:
        print(f"  geom {tag}: scan ({scan[0]:.1f}, {scan[1]:.1f}) R={scan[2]:.1f} | "
              f"bbox ({bbox[0]:.1f}, {bbox[1]:.1f}) R={bbox[2]:.1f} | "
              f"diff {diff:.1f} px{'  <-- CHECK' if flagged else ''}")
    return (*scan, flagged)


# ----------------------------------------------------------------- coordinates

def get_b0(date, time):
    """
    Solar B0 angle (deg) for YYYYMMDD / HHMM. Raises instead of falling back to
    0, since a wrong B0 can shift SRS positions significantly (the offset
    depends on the date and the position on the disk).
    """
    try:
        from astropy.time import Time
        from sunpy.coordinates import sun
    except ImportError as exc:
        raise RuntimeError("B0 needs sunpy and astropy (pip install sunpy)") from exc
    t = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:]}"
    return float(sun.B0(Time(t)).to_value("deg"))


def latlon_to_pixel(lat_deg, lon_deg, r_px, cx, cy, b0_deg=0.0):
    """Heliographic lat/lon -> pixel coordinates. Returns (x, y, visible)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    b0 = np.deg2rad(b0_deg)
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat) * np.cos(b0) - np.cos(lat) * np.cos(lon) * np.sin(b0)
    z = np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(lon) * np.cos(b0)
    return cx + r_px * x, cy + r_px * y, z > 0


def pixel_to_latlon(x, y, r_px, cx, cy, b0_deg=0.0):
    """Inverse of latlon_to_pixel. Returns (lat, lon, on_disk)."""
    X = (np.asarray(x, dtype=float) - cx) / r_px
    Y = (np.asarray(y, dtype=float) - cy) / r_px
    s = X * X + Y * Y
    Z = np.sqrt(np.clip(1.0 - s, 0.0, None))
    b0 = np.deg2rad(b0_deg)
    lat = np.arcsin(np.clip(Y * np.cos(b0) + Z * np.sin(b0), -1.0, 1.0))
    lon = np.arctan2(X, Z * np.cos(b0) - Y * np.sin(b0))
    return np.rad2deg(lat), np.rad2deg(lon), s <= 1.0


def great_circle_deg(lat1, lon1, lat2, lon2):
    """Angular separation on the solar surface, in degrees."""
    a1, o1 = np.deg2rad(np.asarray(lat1, float)), np.deg2rad(np.asarray(lon1, float))
    a2, o2 = np.deg2rad(np.asarray(lat2, float)), np.deg2rad(np.asarray(lon2, float))
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(o1 - o2)
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def mask_to_latlon(mask, r_px, cx, cy, b0_deg, subsample=1):
    """Heliographic coordinates of the on-disk pixels of a boolean mask."""
    ys, xs = np.nonzero(mask)
    if subsample > 1:
        xs, ys = xs[::subsample], ys[::subsample]
    if len(xs) == 0:
        return np.array([]), np.array([])
    lat, lon, on_disk = pixel_to_latlon(xs, ys, r_px, cx, cy, b0_deg)
    return lat[on_disk], lon[on_disk]


def advance_longitudes(regions, hours):
    """Advance SRS longitudes from 00:00 UT by `hours` of solar rotation."""
    shift = ROTATION_DEG_PER_DAY * hours / 24.0
    return [dict(r, lon=r["lon"] + shift) for r in regions]


def is_visible(lat, lon, b0_deg):
    """True if a heliographic position is on the Earth-facing hemisphere."""
    lat, lon, b0 = np.deg2rad(lat), np.deg2rad(lon), np.deg2rad(b0_deg)
    return np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(lon) * np.cos(b0) > 0


# ------------------------------------------------------------------ frame I/O

def frames_for_day(date, fits_root=paths.FITS):
    """Sorted HHMM strings of the magnetograms available for a day."""
    return sorted(p.name.split(".")[0].split("_")[1]
                  for p in (Path(fits_root) / date).glob("*.magnetogram.fits"))


def load_frame(date, time, fits_root=paths.FITS, h5_dir=None, geom_threshold=1.0):
    """
    Load a frame: magnetogram, detection mask (if h5_dir is given), measured disk
    geometry and B0. Returns None if the magnetogram is missing.
    """
    import h5py
    from astropy.io import fits

    stamp = f"{date}_{time}"
    fits_path = Path(fits_root) / date / f"{stamp}.magnetogram.fits"
    if not fits_path.exists():
        return None

    mag = fits.open(fits_path)[1].data
    cx, cy, r_px, flagged = measure_disk(mag, geom_threshold, tag=stamp)

    mask = None
    if h5_dir is not None:
        h5_path = Path(h5_dir) / date / f"{stamp}.magnetogram.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                mask = np.array(f["union_with_intersect"]) > 0

    return {"stamp": stamp, "date": date, "time": time, "mag": mag, "mask": mask,
            "cx": cx, "cy": cy, "r_px": r_px, "b0": get_b0(date, time),
            "flagged": flagged}
