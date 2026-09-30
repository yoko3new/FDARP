"""Regression checks for per-component NOAA distance measurements."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from association import associate_frame
import plot_association
import srs_distance


class AssociationDistanceTest(unittest.TestCase):
    def test_small_component_is_seen_by_statistics_and_plot(self):
        mask = np.zeros((101, 101), dtype=np.uint8)
        mask[50, 10:13] = 1  # Three pixels in the distant component.
        mask[50, 50] = 1     # One pixel at the NOAA position.
        frame = {"mask": mask, "mag": np.zeros_like(mask), "cx": 50,
                 "cy": 50, "r_px": 100, "b0": 0, "time": "0000",
                 "date": "20141020", "stamp": "20141020_0000"}
        region = {"number": "0001", "lat": 0, "lon": 0,
                  "section": "I", "mag": None}

        for subsample in (1, 4):
            with self.subTest(subsample=subsample):
                result = associate_frame(frame, [region], subsample=subsample)
                self.assertEqual(result["counts"], {"1:0": 1, "1:1": 1})
                self.assertEqual(result["nearest"]["0001"], (50, 50, 0.0))

                args = SimpleNamespace(fits_root="", h5_dir="",
                                       subsample=subsample)
                with patch.object(srs_distance, "load_frame", return_value=frame):
                    rows = srs_distance.frame_distances("20141020", "0000",
                                                        [region], args)
                self.assertEqual(rows[0]["dist"], 0.0)

                with tempfile.TemporaryDirectory() as out_dir:
                    args.out_dir = out_dir
                    args.thr_left = 2.0
                    args.thr_right = 3.0
                    args.vlim = 200.0
                    with patch.object(plot_association, "load_frame", return_value=frame):
                        panels = plot_association.make_figure("20141020", "0000",
                                                               [region], args)
                    self.assertEqual(panels[0], ({"I": 1, "IA": 0},
                                                 {"I": 1, "IA": 0}))


if __name__ == "__main__":
    unittest.main()
