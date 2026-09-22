"""Project paths, derived from this file's location."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
FITS = DATA / "fits"            # <YYYYMMDD>/<YYYYMMDD_HHMM>.magnetogram.fits
SRS = DATA / "srs"              # <YYYYMMDD>SRS.txt

RESULTS = ROOT / "results"
DETECTIONS = RESULTS / "detections"   # <variant>/<YYYYMMDD>/*.h5
LOGS = RESULTS / "logs"

FIGURES = ROOT / "figures"
FIG_OVERLAY = FIGURES / "srs_overlay"
FIG_ASSOC = FIGURES / "association"
FIG_GROUPS = FIGURES / "groups"
FIG_DIAG = FIGURES / "diagnostics"

DETECTION_CODE = ROOT / "detection"


def detection_dir(variant="limb"):
    """Output folder of one detection run, e.g. detection_dir('limb')."""
    return DETECTIONS / variant


def srs_file(date):
    """SRS text file for a YYYYMMDD date."""
    return SRS / f"{date}SRS.txt"


def srs_url(date):
    """NOAA archive URL of the SRS file for a YYYYMMDD date."""
    return ("https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
            f"daily_reports/solar_region_summaries/{date[:4]}/{date[4:6]}/"
            f"{date}SRS.txt")


if __name__ == "__main__":
    for name in ("ROOT", "FITS", "SRS", "DETECTIONS", "FIGURES", "DETECTION_CODE"):
        p = globals()[name]
        print(f"{name:<15} {p}  {'(exists)' if p.exists() else '(missing)'}")
