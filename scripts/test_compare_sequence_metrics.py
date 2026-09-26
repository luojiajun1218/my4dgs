import unittest
import json
import tempfile
from pathlib import Path

from scripts.compare_sequence_metrics import (
    RunMetrics,
    common_frame_indices,
    load_run,
    parse_train_log,
    parse_training_metrics,
    render_comparison,
    temporal_windows,
)


def postfix(frame, psnr, size, fps, frame_time, *, closed=True, newline=True):
    record = (
        f"Training progress: 10%|█         | 3/30 [00:03, 0.9it/s, "
        f"Frame index={frame}, Loss=0.01, Loss (Test)=0.02, Num points=60000, "
        f"Active gates=12, Size (MB)={size}, FPS={fps}, PSNR (Test)={psnr}, "
        f"Frame time={frame_time}, Frame time IO=2.5"
    )
    if closed:
        record += "]"
    if newline:
        record += "\n"
    return record


class ParseTrainingMetricsTests(unittest.TestCase):
    def test_json_rows_keep_only_supported_numeric_metrics(self):
        rows = parse_training_metrics([
            {
                "Frame index": 1,
                "PSNR (Test)": "34.25",
                "Size (MB)": 4.07,
                "FPS": 200,
                "Frame time": 9.5,
                "Active gates": 123,
            },
            {"Frame index": "bad", "PSNR (Test)": 10},
            {"Frame index": 2, "PSNR (Test)": float("nan")},
        ])

        self.assertEqual(rows[1], {
            "PSNR (Test)": 34.25,
            "Size (MB)": 4.07,
            "FPS": 200.0,
            "Frame time": 9.5,
        })
        self.assertEqual(rows[2], {})

    def test_log_deduplicates_and_ignores_unclosed_last_postfix(self):
        text = "".join([
            postfix(2, 33.1, 0.4, 220, 9.7),
            postfix(2, 33.4, 0.3, 215, 9.8),
            postfix(3, 33.8, 0.3, 210, 10.0, closed=False, newline=False),
        ])

        rows = parse_train_log(text)

        self.assertEqual(set(rows), {2})
        self.assertEqual(rows[2]["PSNR (Test)"], 33.4)
        self.assertEqual(rows[2]["Size (MB)"], 0.3)
        self.assertEqual(rows[2]["FPS"], 215.0)
        self.assertEqual(rows[2]["Frame time"], 9.8)

    def test_common_frames_and_temporal_windows_use_intersection(self):
        candidate = RunMetrics("candidate", None, {i: {} for i in [1, 2, 10, 31, 120, 121, 150]}, "json", 150, True)
        baseline = RunMetrics("baseline", None, {i: {} for i in [1, 2, 20, 31, 120, 121]}, "json", 121, True)

        common = common_frame_indices(candidate, baseline)

        self.assertEqual(common, [1, 2, 31, 120, 121])
        self.assertEqual(temporal_windows(common), [
            ("2-30", [2]),
            ("31-120", [31, 120]),
            ("121-121", [121]),
        ])


class LoadRunTests(unittest.TestCase):
    def _write_cfg(self, root: Path, *, max_frames: int, resume: str = "None"):
        (root / "cfg_args").write_text(
            "Namespace(max_frames=" + str(max_frames)
            + ", resume_frame1_snapshot=" + resume
            + ", stop_after_frame1_snapshot=False)"
        )

    @staticmethod
    def _json_row(frame, psnr):
        return {
            "Frame index": frame,
            "PSNR (Test)": psnr,
            "Size (MB)": 0.123456,
            "FPS": 200.123456,
            "Frame time": 9.876543,
        }

    def test_resumed_json_is_complete_from_frame_two_and_beats_log_precision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cfg(root, max_frames=3, resume="'/snapshot.pt'")
            rows = [self._json_row(2, 34.2567), self._json_row(3, 35.6789)]
            (root / "training_metrics.json").write_text(json.dumps(rows))
            (root / "train.log").write_text("".join([
                postfix(2, 34.2, 0.12, 200, 9.8),
                postfix(3, 35.6, 0.12, 200, 9.8),
            ]))

            run = load_run(root)

            self.assertTrue(run.complete)
            self.assertEqual(run.expected_start_frame, 2)
            self.assertEqual(set(run.frames), {2, 3})
            self.assertTrue(run.source.endswith("training_metrics.json"))
            self.assertEqual(run.frames[2]["PSNR (Test)"], 34.2567)

            baseline = RunMetrics(
                "baseline", root, {1: {}, 2: {}, 3: {}}, "baseline.json", 3, True
            )
            report = render_comparison(run, baseline)
            self.assertIn("coverage 2-3; expected 2-3; missing expected frames none", report)
            self.assertIn("Aligned frame IDs: 2-3 (2 common frames)", report)

    def test_incomplete_json_falls_back_to_complete_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cfg(root, max_frames=3)
            (root / "training_metrics.json").write_text(json.dumps([
                self._json_row(1, 34.25),
                self._json_row(2, 34.26),
            ]))
            (root / "train.log").write_text("".join([
                postfix(1, 34.2, 0.12, 200, 9.8),
                postfix(2, 34.3, 0.12, 200, 9.8),
                postfix(3, 34.4, 0.12, 200, 9.8),
            ]))

            run = load_run(root)

            self.assertTrue(run.complete)
            self.assertTrue(run.source.endswith("train.log"))
            self.assertEqual(set(run.frames), {1, 2, 3})

    def test_json_missing_required_metric_falls_back_to_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cfg(root, max_frames=2)
            incomplete_row = self._json_row(2, 34.25)
            del incomplete_row["FPS"]
            (root / "training_metrics.json").write_text(json.dumps([
                self._json_row(1, 34.24),
                incomplete_row,
            ]))
            (root / "train.log").write_text("".join([
                postfix(1, 34.2, 0.12, 200, 9.8),
                postfix(2, 34.3, 0.12, 200, 9.8),
            ]))

            run = load_run(root)

            self.assertTrue(run.complete)
            self.assertTrue(run.source.endswith("train.log"))
            self.assertEqual(set(run.frames), {1, 2})

    def test_partial_json_without_log_reports_missing_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cfg(root, max_frames=4, resume="'/snapshot.pt'")
            (root / "training_metrics.json").write_text(json.dumps([
                self._json_row(2, 34.25),
                self._json_row(4, 34.26),
            ]))

            run = load_run(root)
            baseline = RunMetrics(
                "baseline", root, {2: {}, 4: {}}, "baseline.json", 4, False, 2
            )
            report = render_comparison(run, baseline)

            self.assertFalse(run.complete)
            self.assertTrue(run.source.endswith("training_metrics.json"))
            self.assertIn("coverage 2,4; expected 2-4; missing expected frames 3", report)


if __name__ == "__main__":
    unittest.main()
