from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from smp_meso_bindings import (
    BatchExecutionError,
    build_binary,
    compare_transfer_variants,
    decode_float64,
    make_transfer_request,
    run_transfer,
    run_transfer_batch,
    run_transfer_batch_parallel,
)
from smp_meso_bindings.transfer_runner import prepare_transfer_request


def request() -> dict:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "examples/transfer.json").read_text())
    result["max_steps"] = 4
    result["snapshots"]["record_steps"] = np.array([0, 1, 4])
    result["initial"]["probabilities"] = np.empty(0)
    return result


class TransferBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = build_binary(
            command_name="transfer",
            output_path=Path(cls.temporary.name) / "smp-transfer",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def assert_paths_equal(self, left: dict, right: dict) -> None:
        self.assertEqual(left["summaries"], right["summaries"])
        np.testing.assert_array_equal(left["axis"], right["axis"])
        self.assertEqual(len(left["paths"]), len(right["paths"]))
        for a, b in zip(left["paths"], right["paths"]):
            for key in a:
                if key == "elapsed_seconds":
                    continue
                if key == "snapshots":
                    self.assertEqual(a[key].keys(), b[key].keys())
                    for name in a[key]:
                        np.testing.assert_array_equal(a[key][name], b[key][name])
                else:
                    self.assertEqual(a[key], b[key])

    def test_decoding_parallel_order_progress_and_pairing(self) -> None:
        a, b = request(), request()
        a["request_id"], b["request_id"] = "first", "second"
        a["workers"] = 1
        b["workers"] = 4
        events: list[dict] = []
        serial = run_transfer_batch(
            self.binary, [a, b], progress=events.append, progress_step_interval=1
        )
        parallel = run_transfer_batch_parallel(self.binary, [a, b], 2)
        self.assertEqual([r["request_id"] for r in parallel], ["first", "second"])
        for x, y in zip(serial, parallel):
            self.assert_paths_equal(x["result"], y["result"])
        self.assert_paths_equal(serial[0]["result"], serial[1]["result"])
        self.assertTrue(any(e["event"] == "path_heartbeat" for e in events))
        for path in serial[0]["result"]["paths"]:
            self.assertEqual(path["snapshots"]["rho"].shape, (3, 15))
            self.assertEqual(path["snapshots"]["final_edge"].shape, (15, 15))
            self.assertNotIn("edge", path["snapshots"])
            np.testing.assert_allclose(path["snapshots"]["rho"].sum(axis=1), 1)
        # Array preparation must not alter the caller's NumPy request.
        np.testing.assert_array_equal(a["snapshots"]["record_steps"], [0, 1, 4])

    def test_recoverable_errors_and_explicit_fields(self) -> None:
        valid = request()
        invalid = deepcopy(valid)
        invalid["variants"][0]["evolution"] = "D0"
        responses = run_transfer_batch(self.binary, [invalid, valid], check=False)
        self.assertIn("error", responses[0])
        self.assertIn("result", responses[1])
        with self.assertRaises(BatchExecutionError):
            run_transfer(self.binary, invalid)
        missing = deepcopy(valid)
        del missing["snapshots"]["final_score"]
        self.assertIn("error", run_transfer(self.binary, missing, check=False))
        # Large uint64 seeds remain JSON integers, never float64 array values.
        valid["seeds"][0]["init_seed"] = (1 << 64) - 1
        result = run_transfer(self.binary, valid)
        self.assertEqual(result["result"]["paths"][0]["init_seed"], (1 << 64) - 1)

    def test_conversion_and_encoded_inputs(self) -> None:
        item = request()
        lifted = deepcopy(item)
        lifted.update(
            layer="topology",
            paths=200,
            seed=123,
            fast_slow={"mode": "conditional_absorption"},
        )
        converted = make_transfer_request(
            lifted,
            variants=item["variants"],
            seeds=item["seeds"],
            snapshots=item["snapshots"],
        )
        self.assertNotIn("layer", converted)
        self.assertNotIn("paths", converted)
        self.assertNotIn("fast_slow", converted)
        converted["variants"][0]["layer"] = "rho_edge"
        self.assertEqual(item["variants"][0]["layer"], "naive")
        prepared = prepare_transfer_request(converted)
        np.testing.assert_array_equal(
            decode_float64(prepared["snapshots"]["record_steps"]), [0, 1, 4]
        )
        result = run_transfer(self.binary, prepared)
        self.assertEqual(result["result"]["paths"][0]["layer"], "naive")

    def test_comparison_distinguishes_paths_from_marginals(self) -> None:
        result = run_transfer(self.binary, request())
        # Two swaps leave the marginal probability vector unchanged.
        for path in result["result"]["paths"]:
            if path["evolution"] != "deterministic":
                continue
            swap = path["layer"] == "base"
            path["category"] = "k1" if (path["replicate"] == 0) != swap else "k2"
        comparison = compare_transfer_variants(
            result, ("naive", "deterministic"), ("base", "deterministic")
        )
        self.assertEqual(comparison["total_variation"], 0)
        self.assertEqual(comparison["disagreement"], 1)
        self.assertEqual(comparison["pairs"], 2)
        for criterion in ("point_hit", "final_point"):
            compare_transfer_variants(
                result,
                ("naive", "deterministic"),
                ("base", "deterministic"),
                criterion=criterion,
            )
        result["result"]["paths"][1]["initial_hash"] = "different"
        with self.assertRaises(ValueError):
            compare_transfer_variants(
                result, ("naive", "deterministic"), ("base", "deterministic")
            )


if __name__ == "__main__":
    unittest.main()
