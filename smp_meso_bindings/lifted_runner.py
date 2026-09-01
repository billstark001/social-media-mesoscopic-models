"""Python orchestration for the finite-N lifted generator."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from smp_meso_bindings.runner_utils import (
    ProgressCallback,
    run_batch_process,
    run_parallel_processes,
)


def run_lifted_batch(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    return run_batch_process(
        binary_path,
        requests,
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )


def run_lifted_batch_parallel(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    processes: int,
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    return run_parallel_processes(
        binary_path,
        requests,
        processes,
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )


def run_lifted(
    binary_path: os.PathLike[str] | str,
    request: Mapping[str, Any],
    **options: Any,
) -> dict[str, Any]:
    return run_lifted_batch(binary_path, [request], **options)[0]
