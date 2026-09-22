# FDARP Detection

Full-disk active region / PIL detection on HMI magnetograms. Adapted from
SuryaBench `ar_segmentation`; see `LICENSE-upstream`.

## Build

Requires a C++17 compiler, CMake, OpenCV (core and imgproc), HDF5 with C++
support, yaml-cpp, and CFITSIO.

From the `detection` directory:

```sh
cmake -S . -B build
cmake --build build -j 4
```

### macOS SDK compatibility

If linking fails with `tapi error: malformed file` and
`unknown architecture arm64e.x1` on the macOS 27 SDK, pass an installed SDK
that your linker supports on the command line (do not hard-code it in
`CMakeLists.txt`):

```sh
cmake --fresh -S . -B build \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build -j 4
```

`--fresh` requires CMake 3.24+; with older versions, use a new build directory.

## Run

Run from the `detection` directory (the program reads `config.yaml` from the
current directory, and paths in it are relative to `detection/`):

```sh
./build/fdarp_detect 20141020
```

- Input: `../data/fits/<YYYYMMDD>/*.magnetogram.fits`
- Output: `../results/detections/limb/<YYYYMMDD>/*.h5`, with datasets
  `union_with_intersect` (AR mask) and `intersection` (PIL region)
- Existing output files are skipped.

## Configuration

See `config.yaml`. The limb cut is set by `limb_mode` and `limb_value`:

| limb_mode     | cut radius                          |
|---------------|-------------------------------------|
| `none`        | no cut                              |
| `radius_frac` | `limb_value * R_sun` (default 1.0)  |
| `radius_px`   | `limb_value` pixels                 |
| `angle`       | `R_sun * sin(limb_value deg)`       |
| `formula`     | measured `R_sun` + 900 / 0.6 px (upstream formula; larger than the frame) |

The disk centre and radius are measured from each frame (`measure_disk: true`).
Build with `-DFDARP_DEBUG=ON` for per-step output.