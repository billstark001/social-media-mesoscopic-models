"""Binary-array request and response orchestration for the kinetic solver."""

from __future__ import annotations

import base64
import copy
import os
import zlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from smp_meso_bindings.runner_utils import (
    ProgressCallback,
    run_batch_process,
    run_parallel_processes,
)

FLOAT64_ENCODING = "base64+zlib+f64le"


def encode_float64(values: ArrayLike) -> dict[str, str]:
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim == 0:
        array = array.reshape(1)
    shape = "x".join(str(value) for value in array.shape)
    data = base64.b64encode(zlib.compress(array.tobytes(order="C"), level=9))
    return {"encoding": FLOAT64_ENCODING, "shape": shape, "data": data.decode("ascii")}


def decode_float64(payload: Mapping[str, Any]) -> NDArray[np.float64]:
    if payload.get("encoding") != FLOAT64_ENCODING:
        raise ValueError(f"unsupported array encoding {payload.get('encoding')!r}")
    try:
        shape = tuple(int(value) for value in str(payload["shape"]).split("x"))
    except (KeyError, ValueError) as error:
        raise ValueError("invalid encoded array shape") from error
    if not shape or any(value < 0 for value in shape):
        raise ValueError("invalid encoded array shape")
    raw = zlib.decompress(base64.b64decode(str(payload["data"]), validate=True))
    expected = int(np.prod(shape, dtype=np.int64))
    result = np.frombuffer(raw, dtype="<f8")
    if result.size != expected:
        raise ValueError(f"encoded array contains {result.size} values, expected {expected}")
    return result.reshape(shape).copy()


def prepare_kinetic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a request and encode every numerical array not encoded yet."""

    result = copy.deepcopy(dict(request))
    initial = result.get("initial")
    if isinstance(initial, dict):
        probabilities = initial.get("probabilities")
        if not isinstance(probabilities, Mapping):
            initial["probabilities"] = encode_float64(probabilities)
    snapshots = result.get("snapshots")
    if isinstance(snapshots, dict):
        record_steps = snapshots.get("record_steps")
        if not isinstance(record_steps, Mapping):
            snapshots["record_steps"] = encode_float64(record_steps)
    return result


def decode_kinetic_response(response: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(response))
    numerical = result.get("result")
    if not isinstance(numerical, dict):
        return result
    series = numerical.get("series")
    if isinstance(series, dict):
        for name, value in tuple(series.items()):
            if isinstance(value, Mapping):
                series[name] = decode_float64(value)
    snapshots = numerical.get("snapshots")
    if isinstance(snapshots, dict):
        for name, value in tuple(snapshots.items()):
            if isinstance(value, Mapping) and value.get("encoding") == FLOAT64_ENCODING:
                snapshots[name] = decode_float64(value)
    return result


def run_kinetic_batch(
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
        (prepare_kinetic_request(request) for request in requests),
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )
    return [decode_kinetic_response(response) for response in responses]


def run_kinetic_batch_parallel(
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
        (prepare_kinetic_request(request) for request in requests),
        processes,
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )
    return [decode_kinetic_response(response) for response in responses]


def run_kinetic(
    binary_path: os.PathLike[str] | str,
    request: Mapping[str, Any],
    **options: Any,
) -> dict[str, Any]:
    return run_kinetic_batch(binary_path, [request], **options)[0]
