"""Single- and multi-process JSONL orchestration for the Go solver."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


class BatchExecutionError(RuntimeError):
    """One or more Go requests returned a structured error."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        failures = [item for item in responses if item.get("error")]
        message = "; ".join(
            f"{item.get('request_id', '<unknown>')}: {item['error']}"
            for item in failures
        )
        super().__init__(message)


def _binary_path(binary_path: os.PathLike[str] | str) -> str:
    path = Path(binary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"binary is not executable: {path}")
    return str(path)


def _payload(requests: list[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(request), separators=(",", ":"), allow_nan=False) + "\n"
        for request in requests
    )


def _validate_progress_interval(
    progress: ProgressCallback | None,
    progress_step_interval: int,
) -> None:
    if progress_step_interval < 0:
        raise ValueError("progress_step_interval must be nonnegative")
    if progress is None and progress_step_interval != 0:
        raise ValueError("progress_step_interval requires a progress callback")


def _parse_responses(
    stdout_lines: list[str],
    expected: int,
    returncode: int,
    stderr_lines: list[str],
) -> list[dict[str, Any]]:
    if returncode != 0:
        stderr = "".join(stderr_lines).strip()
        raise RuntimeError(f"smp-meso batch exited with {returncode}: {stderr}")
    responses = [json.loads(line) for line in stdout_lines if line.strip()]
    if len(responses) != expected:
        raise RuntimeError(f"expected {expected} responses, received {len(responses)}")
    return responses


def _run_streaming(
    binary: str,
    request_list: list[Mapping[str, Any]],
    environment: Mapping[str, str] | None,
    progress: ProgressCallback,
    progress_step_interval: int,
) -> list[dict[str, Any]]:
    process_environment = dict(os.environ)
    if environment:
        process_environment.update(environment)
    command = [binary, "batch", "--progress", "jsonl"]
    if progress_step_interval:
        command.extend(["--progress-step-interval", str(progress_step_interval)])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=process_environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process_stdin = process.stdin
    process_stdout = process.stdout
    process_stderr = process.stderr
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    callback_errors: list[Exception] = []

    def read_stdout() -> None:
        stdout_lines.extend(process_stdout)

    def read_stderr() -> None:
        for line in process_stderr:
            stderr_lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if callback_errors:
                continue
            try:
                progress(event)
            except Exception as error:  # noqa: BLE001 - preserve caller exception
                callback_errors.append(error)

    stdout_thread = threading.Thread(target=read_stdout, name="smp-meso-stdout")
    stderr_thread = threading.Thread(target=read_stderr, name="smp-meso-progress")
    stdout_thread.start()
    stderr_thread.start()
    try:
        process_stdin.write(_payload(request_list))
        process_stdin.close()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        stdout_thread.join()
        stderr_thread.join()
        process_stdout.close()
        process_stderr.close()
    if callback_errors:
        raise callback_errors[0]
    return _parse_responses(stdout_lines, len(request_list), returncode, stderr_lines)


def _run_quiet(
    binary: str,
    request_list: list[Mapping[str, Any]],
    environment: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    process_environment = dict(os.environ)
    if environment:
        process_environment.update(environment)
    completed = subprocess.run(
        [binary, "batch"],
        input=_payload(request_list),
        text=True,
        capture_output=True,
        env=process_environment,
        check=False,
    )
    return _parse_responses(
        completed.stdout.splitlines(keepends=True),
        len(request_list),
        completed.returncode,
        completed.stderr.splitlines(keepends=True),
    )


def _check_responses(
    responses: list[dict[str, Any]],
    check: bool,
) -> list[dict[str, Any]]:
    if check and any(response.get("error") for response in responses):
        raise BatchExecutionError(responses)
    return responses


def run_batch(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate multiple parameter points in one clean Go process.

    With no ``progress`` callback, the subprocess uses the quiet compatibility
    path. With a callback, stdout and stderr are drained concurrently and each
    Go JSONL progress event is delivered in real time. No numerical request is
    mutated by the binding.
    """

    request_list = list(requests)
    if not request_list:
        return []
    _validate_progress_interval(progress, progress_step_interval)
    binary = _binary_path(binary_path)
    if progress is None:
        responses = _run_quiet(binary, request_list, environment)
    else:
        total = len(request_list)

        def enrich(event: dict[str, Any]) -> None:
            item = dict(event)
            item["request_index"] = int(item.get("batch_index", 0))
            item["request_total"] = total
            item["process_index"] = 1
            item["process_total"] = 1
            progress(item)

        responses = _run_streaming(
            binary, request_list, environment, enrich, progress_step_interval
        )
    return _check_responses(responses, check)


def run_batch_parallel(
    binary_path: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    processes: int,
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate a scan with ``processes`` long-lived Go batch processes.

    Each process takes its next request from a shared queue only after producing
    the preceding response. This dynamic scheduling matters when absorption
    times vary widely across parameter points. Responses are restored to input
    order. Each request's explicit ``workers`` value still controls path-level
    goroutines inside its Go process, so callers should budget roughly
    ``processes * workers`` runnable paths.
    """

    request_list = list(requests)
    if not request_list:
        return []
    if processes < 1:
        raise ValueError("processes must be positive")
    if processes > len(request_list):
        raise ValueError("processes must not exceed the number of requests")
    _validate_progress_interval(progress, progress_step_interval)
    binary = _binary_path(binary_path)
    pending: queue.Queue[tuple[int, Mapping[str, Any]]] = queue.Queue()
    for index, request in enumerate(request_list):
        pending.put((index, request))
    ordered: list[dict[str, Any] | None] = [None] * len(request_list)
    callback_lock = threading.Lock()
    stop = threading.Event()

    def run_worker(process_index: int) -> list[tuple[int, dict[str, Any]]]:
        process_environment = dict(os.environ)
        if environment:
            process_environment.update(environment)
        command = [binary, "batch"]
        if progress is not None:
            command.extend(["--progress", "jsonl"])
            if progress_step_interval:
                command.extend(
                    ["--progress-step-interval", str(progress_step_interval)]
                )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=process_environment,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process_stdin = process.stdin
        process_stdout = process.stdout
        process_stderr = process.stderr
        stderr_lines: list[str] = []
        callback_errors: list[Exception] = []
        local_to_global: dict[int, int] = {}
        mapping_lock = threading.Lock()

        def read_stderr() -> None:
            for line in process_stderr:
                stderr_lines.append(line)
                if progress is None or callback_errors:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local_index = int(event.get("batch_index", 0))
                with mapping_lock:
                    global_index = local_to_global.get(local_index, -1)
                item = dict(event)
                item["request_index"] = global_index + 1
                item["request_total"] = len(request_list)
                item["process_index"] = process_index + 1
                item["process_total"] = processes
                try:
                    with callback_lock:
                        progress(item)
                except Exception as error:  # noqa: BLE001 - preserve caller exception
                    callback_errors.append(error)

        stderr_thread = threading.Thread(
            target=read_stderr,
            name=f"smp-meso-progress-{process_index + 1}",
        )
        stderr_thread.start()
        completed: list[tuple[int, dict[str, Any]]] = []
        returncode: int | None = None
        local_index = 0
        try:
            while not stop.is_set():
                try:
                    global_index, request = pending.get_nowait()
                except queue.Empty:
                    break
                local_index += 1
                with mapping_lock:
                    local_to_global[local_index] = global_index
                process_stdin.write(_payload([request]))
                process_stdin.flush()
                response_line = process_stdout.readline()
                if not response_line:
                    returncode = process.poll()
                    detail = "".join(stderr_lines).strip()
                    raise RuntimeError(
                        "smp-meso batch ended before returning request "
                        f"{global_index + 1}; exit={returncode}: {detail}"
                    )
                completed.append((global_index, json.loads(response_line)))
            process_stdin.close()
            returncode = process.wait()
            if returncode != 0:
                detail = "".join(stderr_lines).strip()
                raise RuntimeError(f"smp-meso batch exited with {returncode}: {detail}")
        except BaseException:
            stop.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if not process_stdin.closed:
                try:
                    process_stdin.close()
                except BrokenPipeError:
                    pass
            stderr_thread.join()
            process_stdout.close()
            process_stderr.close()
        if callback_errors:
            stop.set()
            raise callback_errors[0]
        return completed

    with ThreadPoolExecutor(
        max_workers=processes, thread_name_prefix="smp-meso-go"
    ) as pool:
        futures = [
            pool.submit(run_worker, process_index) for process_index in range(processes)
        ]
        for future in futures:
            for index, response in future.result():
                ordered[index] = response
    responses = [response for response in ordered if response is not None]
    if len(responses) != len(request_list):
        raise RuntimeError("parallel result assembly lost one or more responses")
    return _check_responses(responses, check)


def print_progress(event: Mapping[str, Any]) -> None:
    """Compact stderr printer suitable as the bindings' progress callback."""

    request = event.get("request_id", "<unknown>")
    location = f"[{event.get('request_index', '?')}/{event.get('request_total', '?')}"
    if int(event.get("process_total", 1)) > 1:
        location += f" p{event['process_index']}"
    location += "]"
    kind = event.get("event", "progress")
    if kind == "path_heartbeat":
        scenario = ""
        if int(event.get("scenario_index", 0)) > 0:
            scenario = (
                f" scenario {event.get('scenario_index')}/{event.get('scenario_count')}"
            )
        detail = (
            f"{event.get('stage')}{scenario} path {event.get('path_index')}/"
            f"{event.get('total_paths')} step {event.get('step')}"
        )
    elif kind == "path_completed":
        detail = (
            f"{event.get('stage')} {event.get('completed_paths')}/"
            f"{event.get('total_paths')} paths; last={event.get('category')} "
            f"step={event.get('step')}"
        )
    elif kind.startswith("scenario_"):
        detail = f"scenario {event.get('scenario_index')}/{event.get('scenario_count')}"
    elif kind in {"request_rejected", "request_failed"}:
        detail = f"{kind}: {event.get('message', '<no detail>')}"
    else:
        detail = kind
    elapsed = float(event.get("elapsed_seconds", 0.0))
    print(
        f"{location} {request}: {detail} ({elapsed:.1f}s)",
        file=sys.stderr,
        flush=True,
    )


def run_one(
    binary_path: os.PathLike[str] | str,
    request: Mapping[str, Any],
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    progress: ProgressCallback | None = None,
    progress_step_interval: int = 0,
) -> dict[str, Any]:
    """Evaluate one parameter point through the same single-process protocol."""

    return run_batch(
        binary_path,
        [request],
        check=check,
        environment=environment,
        progress=progress,
        progress_step_interval=progress_step_interval,
    )[0]
