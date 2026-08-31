from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path

from smp_meso_bindings import (
    build_binary,
    print_progress,
    run_batch,
    run_batch_parallel,
)


def request(request_id: str, layer: str, seed: int) -> dict[str, object]:
    return {
        "request_id": request_id,
        "layer": layer,
        "population": 30,
        "opinion_bins": 5,
        "out_degree": 3,
        "recommendation_count": 2,
        "max_steps": 3,
        "paths": 2,
        "interval_paths": 1,
        "ambiguity_samples": 2,
        "confidence_level": 0.95,
        "workers": 1,
        "seed": seed,
        "major_cluster_mass": 0.05,
        "dynamics": {
            "type": "hk",
            "tolerance": 0.45,
            "influence": 0.05,
            "rewiring_rate": 0.02,
        },
        "recommender": {
            "type": "random",
            "steepness": 0.0,
            "random_ratio": 0.0,
            "opinion_tolerance": 0.4,
            "noise_std": 0.0,
            "noise_quadrature_points": 1,
        },
        "initial": {
            "type": "uniform",
            "opinion_min": -1.0,
            "opinion_max": 1.0,
            "probabilities": [],
        },
        "resolution": {
            "score_max": 5,
            "availability_bins": 3,
            "component_size_bins": 3,
            "opinion_quadrature_points": 1,
        },
        "closure": {
            "motif_relaxation": 0.2,
            "histogram_relaxation": 0.2,
            "candidate_relaxation": 0.2,
            "topology_relaxation": 0.2,
        },
        "ambiguity": {
            "eligibility_correlation_radius": 0.2,
            "score_availability_radius": 0.2,
            "motif_persistence_radius": 0.2,
            "bridge_bias_radius": 0.2,
            "component_mix_radius": 0.2,
        },
    }


class BindingsIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = build_binary(
            source_root=Path(__file__).resolve().parents[1],
            output_path=Path(cls.temporary.name) / "smp-meso",
            backend="purego",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_streamed_and_parallel_results_are_identical_and_ordered(self) -> None:
        requests = [request("one", "base", 11), request("two", "topology", 12)]
        single_events: list[dict[str, object]] = []
        single = run_batch(
            self.binary,
            requests,
            progress=single_events.append,
            progress_step_interval=1,
        )
        parallel_events: list[dict[str, object]] = []
        parallel = run_batch_parallel(
            self.binary,
            requests,
            2,
            progress=parallel_events.append,
            progress_step_interval=1,
        )
        self.assertEqual([item["request_id"] for item in parallel], ["one", "two"])
        for left, right in zip(single, parallel, strict=True):
            self.assertEqual(left["result"]["point"], right["result"]["point"])
            self.assertEqual(left["result"]["interval"], right["result"]["interval"])
        self.assertTrue(
            any(event["event"] == "path_heartbeat" for event in single_events)
        )
        self.assertEqual({event["process_index"] for event in parallel_events}, {1, 2})
        self.assertEqual(
            {
                event["request_index"]
                for event in parallel_events
                if event["event"] == "request_started"
            },
            {1, 2},
        )

    def test_quiet_batch_does_not_require_progress(self) -> None:
        item = request("quiet", "base", 13)
        response = run_batch(self.binary, [deepcopy(item)])[0]
        self.assertEqual(response["request_id"], "quiet")
        self.assertAlmostEqual(sum(response["result"]["point"]["probabilities"]), 1.0)

    def test_parallel_scheduler_assigns_next_request_to_free_process(self) -> None:
        fake = Path(self.temporary.name) / "fake-smp-meso"
        fake.write_text(
            """#!/usr/bin/env python3
import json
import sys
import time

for batch_index, line in enumerate(sys.stdin, 1):
    request = json.loads(line)
    print(json.dumps({
        "event": "request_started",
        "request_id": request["request_id"],
        "batch_index": batch_index,
    }), file=sys.stderr, flush=True)
    time.sleep(request["delay"])
    print(json.dumps({
        "request_id": request["request_id"],
        "result": {"delay": request["delay"]},
    }), flush=True)
""",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        requests = [
            {"request_id": "slow", "delay": 0.2},
            {"request_id": "fast-1", "delay": 0.01},
            {"request_id": "fast-2", "delay": 0.01},
            {"request_id": "fast-3", "delay": 0.01},
        ]
        events: list[dict[str, object]] = []
        responses = run_batch_parallel(fake, requests, 2, progress=events.append)
        self.assertEqual(
            [response["request_id"] for response in responses],
            [request["request_id"] for request in requests],
        )
        started = [event for event in events if event["event"] == "request_started"]
        assignments = Counter(int(event["process_index"]) for event in started)
        self.assertEqual(sorted(assignments.values()), [1, 3])
        self.assertEqual(
            {event["request_id"]: event["request_index"] for event in started},
            {"slow": 1, "fast-1": 2, "fast-2": 3, "fast-3": 4},
        )

    def test_progress_printer_includes_process_and_failure_detail(self) -> None:
        output = StringIO()
        with redirect_stderr(output):
            print_progress(
                {
                    "event": "request_failed",
                    "request_id": "broken",
                    "request_index": 2,
                    "request_total": 4,
                    "process_index": 1,
                    "process_total": 2,
                    "message": "numerical invariant failed",
                }
            )
        rendered = output.getvalue()
        self.assertIn("[2/4 p1]", rendered)
        self.assertIn("numerical invariant failed", rendered)


if __name__ == "__main__":
    unittest.main()
