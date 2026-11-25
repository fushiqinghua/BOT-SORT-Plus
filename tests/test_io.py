import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from botsort_research.io import (
    detection_files, forbid_gt, image_path, load_detections, load_gt,
    selected_sequences, sha256,
)


class CacheIOTests(unittest.TestCase):
    def test_gzip_roundtrip_and_empty_frame_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "seq.json.gz"
            payload = {"frames": [{"frame": 1, "detections": []}, {"frame": 2, "detections": [[1, 2, 3, 4, .9, 6]]}]}
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream)
            self.assertEqual(load_detections(source), payload)
            self.assertEqual(detection_files(root), {"seq": source})
            self.assertEqual(len(sha256(source)), 64)
            (root / "seq.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Both compressed and plain"):
                detection_files(root)

    def test_rejects_skipped_frames_and_bad_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seq.json"
            cases = [
                {"frames": [{"frame": 1, "detections": []}, {"frame": 3, "detections": []}]},
                {"frames": [{"frame": 1, "detections": [[1, 2, 1, 4, .9, 0]]}]},
                {"frames": [{"frame": 1, "detections": [[1, 2, 3, 4, 1.1, 0]]}]},
                {"frames": [{"frame": 1, "detections": [[1, 2, 3, 4, .9]]}]},
            ]
            for payload in cases:
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_detections(path)


class GroundTruthTests(unittest.TestCase):
    def test_pool_classes_keep_native_coordinates_ignore_unmarked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "seq" / "gt" / "gt.txt"
            path.parent.mkdir(parents=True)
            path.write_text("1,7,2,3,4,5,1,6,1\n1,8,4,5,6,7,0,2,1\n2,9,1,2,3,4,1,3,1\n")
            got = load_gt(root, "seq")
            np.testing.assert_array_equal(got[1], [[2, 3, 6, 8, 7]])
            np.testing.assert_array_equal(got[2], [[1, 2, 4, 6, 9]])
            path.write_text("")
            self.assertEqual(load_gt(root, "seq"), {})

    def test_duplicate_identity_fails_and_prediction_guard_blocks_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seq" / "gt" / "gt.txt"
            path.parent.mkdir(parents=True)
            path.write_text("1,7,2,3,4,5,1,1,1\n1,7,4,5,6,7,1,2,1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate GT"):
                load_gt(temp, "seq")
            for p in [path, "C:\\dataset\\seq\\gt\\gt.txt", "/dataset/annotations/data.json"]:
                with self.assertRaises(PermissionError):
                    forbid_gt("open", (p, "r"))
            forbid_gt("open", ("/cache/sequence.json", "r"))


class ImageAndSequenceTests(unittest.TestCase):
    def test_official_suffix_remapping_and_escape_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            expected = root / "test2017/DJI_0001_1/00/00000.jpg"
            self.assertEqual(image_path({"frame": 1, "image": "test2017/DJI_0001_1/00/00000.jpg"}, root), expected)
            self.assertEqual(image_path({"frame": 1, "image": "/old/nonexistent/VT-Tiny-MOT/test2017/DJI_0001_1/00/00000.jpg"}, root), expected)
            with self.assertRaisesRegex(ValueError, "Cannot remap"):
                image_path({"frame": 1, "image": "../secret.jpg"}, root)
            with self.assertRaisesRegex(ValueError, "no thermal"):
                image_path({"frame": 1, "image": "test2017/a.jpg"}, root, "thermal")

    def test_subset_selection_does_not_silently_cross_partition(self):
        files = {"a": Path("a"), "b": Path("b"), "c": Path("c")}
        split = {"calibration": ["a"], "evaluation": ["b", "c"]}
        self.assertEqual(selected_sequences(files, split), ["b", "c"])
        self.assertEqual(selected_sequences(files, split, sequences=["c"]), ["c"])
        with self.assertRaisesRegex(ValueError, "within split"):
            selected_sequences(files, split, sequences=["a"])
        with self.assertRaises(FileNotFoundError):
            selected_sequences(files, sequences=["missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            selected_sequences(files, sequences=["a", "a"])


if __name__ == "__main__":
    unittest.main()
