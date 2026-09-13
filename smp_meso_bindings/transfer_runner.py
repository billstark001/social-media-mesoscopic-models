"""Paired transfer/generator experiments with explicit initialization streams."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np

from smp_meso_bindings.arrays import decode_float64, encode_float64
from smp_meso_bindings.runner_utils import (
    ProgressCallback,
    run_batch_process,
    run_parallel_processes,
)


def make_transfer_request(
    lifted_request: Mapping[str, Any],
    *,
    variants: Iterable[Mapping[str, Any]],
    seeds: Iterable[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy shared physics from a lifted request into the experiment schema.

    The caller explicitly supplies variants, both seed streams and every snapshot
    switch. Lifted layer, seed, ensemble, ambiguity and fast-slow controls do not
    carry over: experiments always use the zero-profile unsplit law. No model
    parameter is inferred and neither input is mutated.
    """

    names = (
        "request_id",
        "population",
        "opinion_bins",
        "out_degree",
        "recommendation_count",
        "max_steps",
        "workers",
        "major_cluster_mass",
        "terminal_position_resolution",
        "terminal_mass_resolution",
        "dynamics",
        "recommender",
        "initial",
    )
    result = copy.deepcopy({name: lifted_request[name] for name in names})
    result["resolution"] = {
        name: lifted_request["resolution"][name]
        for name in (
            "score_max",
            "opinion_quadrature_points",
            "opinion_quadrature_rule",
        )
    }
    result["closure"] = {
        "motif_relaxation": lifted_request["closure"]["motif_relaxation"]
    }
    result["variants"] = copy.deepcopy([dict(value) for value in variants])
    result["seeds"] = copy.deepcopy([dict(value) for value in seeds])
    result["snapshots"] = copy.deepcopy(dict(snapshots))
    return result


def prepare_transfer_request(request: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(request))
    for block, field in (("initial", "probabilities"), ("snapshots", "record_steps")):
        values = result.get(block)
        if (
            isinstance(values, dict)
            and field in values
            and not isinstance(values[field], Mapping)
        ):
            values[field] = encode_float64(values[field])
    return result


def decode_transfer_response(response: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(response))
    numerical = result.get("result")
    if not isinstance(numerical, dict):
        return result
    numerical["axis"] = decode_float64(numerical["axis"])
    for path in numerical["paths"]:
        path["snapshots"] = {
            name: decode_float64(value) for name, value in path["snapshots"].items()
        }
    return result


def run_transfer_batch(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    responses = run_batch_process(
        binary_path,
        (prepare_transfer_request(request) for request in requests),
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )
    return [decode_transfer_response(response) for response in responses]


def run_transfer_batch_parallel(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    processes: int,
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    responses = run_parallel_processes(
        binary_path,
        (prepare_transfer_request(request) for request in requests),
        processes,
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )
    return [decode_transfer_response(response) for response in responses]


def run_transfer(
    binary_path: os.PathLike[str] | str,
    request: Mapping[str, Any],
    **options: Any,
) -> dict[str, Any]:
    return run_transfer_batch(binary_path, [request], **options)[0]


def compare_transfer_variants(
    response: Mapping[str, Any],
    left: tuple[str, str],
    right: tuple[str, str],
    *,
    criterion: Literal["primary", "point_hit", "final_point"] = "primary",
) -> dict[str, Any]:
    """Compare (canonical layer, evolution) pairs within one successful response.

    Return the paired category confusion matrix, disagreement rate, marginal
    probability vectors and their total-variation distance. These descriptive
    statistics do not assert equivalence or independent sampling. Repeated
    initial seeds (conditional continuations) remain separate requested pairs.
    """

    if criterion not in {"primary", "point_hit", "final_point"}:
        raise ValueError(f"unsupported criterion {criterion!r}")
    result = response.get("result")
    if not isinstance(result, Mapping) or response.get("error"):
        raise ValueError("comparison requires a successful transfer response")
    categories = list(result["categories"])
    groups: list[dict[int, Mapping[str, Any]]] = []
    for variant in (left, right):
        paths = [
            path
            for path in result["paths"]
            if (path["layer"], path["evolution"]) == variant
        ]
        group = {path["replicate"]: path for path in paths}
        if not group or len(group) != len(paths):
            raise ValueError(f"missing or duplicate paths for variant {variant!r}")
        groups.append(group)
    if groups[0].keys() != groups[1].keys():
        raise ValueError("variants have different replicate identifiers")

    def category(path: Mapping[str, Any]) -> str:
        if criterion == "primary":
            return str(path["category"])
        if criterion == "point_hit":
            hit = path["point_hit"]
            return str(hit["category"]) if hit is not None else "censored"
        final = path["final_point"]
        return str(final["category"]) if final["status"] == "absorbed" else "censored"

    confusion = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for replicate, a in groups[0].items():
        b = groups[1][replicate]
        if a["initial_hash"] != b["initial_hash"] or a["init_seed"] != b["init_seed"]:
            raise ValueError(f"initial states differ for replicate {replicate}")
        confusion[categories.index(category(a)), categories.index(category(b))] += 1
    count = len(groups[0])
    left_probabilities = confusion.sum(axis=1) / count
    right_probabilities = confusion.sum(axis=0) / count
    return {
        "categories": categories,
        "pairs": count,
        "criterion": criterion,
        "confusion": confusion,
        "left_probabilities": left_probabilities,
        "right_probabilities": right_probabilities,
        "disagreement": float(1 - np.trace(confusion) / count),
        "total_variation": float(
            np.abs(left_probabilities - right_probabilities).sum() / 2
        ),
    }
