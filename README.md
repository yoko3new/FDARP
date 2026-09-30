# FDARP

Full-disk active region (AR) and polarity inversion line (PIL) detection on
Surya-bench HMI line-of-sight magnetograms, with association to NOAA Solar
Region Summary (SRS) regions.

## Layout

```
detection/   C++ AR / PIL detection (adapted from SuryaBench ar_segmentation)
pipeline/    Python: data download, SRS parsing, association, figures
data/        magnetograms and SRS files (not tracked)
results/     detection output (not tracked)
figures/     generated figures (not tracked)
```

## Requirements

- C++ build: see `detection/README.md`
- Python 3.10+: `pip install -r requirements.txt` (versions in the file are the tested lower bounds)
- AWS CLI, used to download Surya-bench (public bucket, no credentials)

## Quick start

One day, hourly frames. Build the detection code first (from the repository
root):

```sh
cd detection
cmake -S . -B build
cmake --build build -j 4
cd ..
```

On macOS, see `detection/README.md` if linking fails. Then:

```sh
# 1. data
cd pipeline
python download_srs.py --date 20141020
python download_extract.py --start 20141020 --end 20141020 --hours 1

# 2. detection
cd ../detection
./build/fdarp_detect 20141020

# 3. association and figures
cd ../pipeline
python check_groups.py --date 20141020 --all
python srs_overlay.py --date 20141020 --all
python plot_association.py --date 20141020 --all
```

The Python scripts accept dates as `YYYYMMDD` or `YYYY-MM-DD`; `fdarp_detect`
takes `YYYYMMDD` (the name of the data folder).

## Pipeline scripts

| Script | Purpose |
|---|---|
| `download_extract.py` | Download Surya-bench `.nc`, keep `hmi_m` as FITS |
| `download_srs.py` | Download NOAA SRS files |
| `check_groups.py` | Classify component / NOAA-region groups for a day |
| `batch_days.py` | Same over many days, one frame per day |
| `srs_distance.py` | Distance from each SRS region to the nearest detection |
| `srs_overlay.py` | Figure: detections with SRS regions |
| `plot_association.py` | Figure: association at two thresholds |
| `scan_srs.py` | Find days with close NOAA region pairs |
| `solar_utils.py`, `association.py`, `paths.py` | Shared modules |

## Method notes

- **Disk geometry** is measured from each magnetogram. In Surya-bench frames
  the disk centre is not at the image centre.
- **Limb cut**: at the solar limb by default (`limb_mode: radius_frac`,
  `limb_value: 1.0` in `detection/config.yaml`).
- **Association**: an SRS region is associated with a detected component if the
  great-circle distance on the solar surface is at most 3 degrees. Distances
  are in degrees, not pixels, because foreshortening makes a pixel distance
  mean different things across the disk.
- **SRS positions** are valid at 00:00 UT; for later frames longitudes are
  advanced at 13.2 deg/day. The B0 angle is computed per frame with SunPy.
- **NOAA numbers**: SRS reports show only four digits. For the validated
  2012-2015 archive, the parser adds 10000 and keeps both the original
  `srs_number` (including leading zeros) and full integer `noaa_ar`, along
  with `srs_date`. The existing `number` field is the full number as a string,
  so figures and tables display full IDs. Dates outside this range fail until
  their numbering epoch is checked. See the [NOAA SRS format description](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/docs/SolarRegionSummary.pdf).
- **Groups**: `1:1`, `1:N` (one component, several regions), `N:1`, `M:N`,
  `1:0` (component without a NOAA region), `0:1` (region not detected).
- **Future tracking**: use a separate `track_id` for detected trajectories and
  store all associated NOAA IDs for each observation. One detected component
  can cover more than one NOAA region; its trajectory should not be split
  solely to force a one-to-one NOAA mapping.

## License

- Original code in this repository (`pipeline/` and repository-level files):
  MIT, see `LICENSE`.
- `detection/` is adapted from SuryaBench `ar_segmentation` and is licensed as
  declared by upstream; see `detection/LICENSE-upstream` and the file headers.
  Upstream copyright notices are retained.
