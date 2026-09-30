"""Check lossless SRS IDs and the calibrated NOAA numbering epoch."""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from solar_utils import parse_srs, resolve_noaa_ar


class SrsNumberTest(unittest.TestCase):
    def test_full_ids_and_original_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "20141020SRS.txt"
            path.write_text(":Issued: 2014 Oct 20 0030 UTC\n"
                            "I. Regions with Sunspots\n"
                            "2192 S13E43 251 1560 Fkc 22 32 Beta-Gamma-Delta\n"
                            "IA. H-alpha Plages without Spots.\n"
                            "2190 N23W09 306\n"
                            "II. Regions Due to Return\n"
                            "2177 N13 191\n")
            regions = parse_srs(path)

        self.assertEqual([r["number"] for r in regions], ["12192", "12190"])
        self.assertEqual([r["noaa_ar"] for r in regions], [12192, 12190])
        self.assertEqual([r["srs_number"] for r in regions], ["2192", "2190"])
        self.assertEqual([r["srs_date"] for r in regions], ["20141020"] * 2)

    def test_number_epoch_across_archive(self):
        self.assertEqual(resolve_noaa_ar("1429", date(2012, 3, 7)), 11429)
        self.assertEqual(resolve_noaa_ar("2473", date(2015, 12, 31)), 12473)

    def test_uncalibrated_date_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            resolve_noaa_ar("2192", date(2020, 1, 1))


if __name__ == "__main__":
    unittest.main()
