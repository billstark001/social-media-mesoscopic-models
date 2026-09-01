"""Process and JSONL utilities shared by the lifted and kinetic runners."""

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
        super().__init__(
            "; ".join(
                f"{item.get('request_id', '<unknown>')}: {item['error']}"
                for item in failures
            )
        )


def binary_path(binary_path: os.PathLike[str] | str) -> str:
    path = Path(binary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"binary is not executable: {path}")
    return str(path)


def payload(requests: list[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(request), separators=(",", ":"), allow_nan=False) + "\n"
        for request in requests
    )


def validate_progress(
    progress: ProgressCallback | None, progress_step_interval: int
) -> None:
    if progress_step_interval < 0:
        raise ValueError("progress_step_interval must be nonnegative")
    if progress is None and progress_step_interval != 0:
        raise ValueError("progress_step_interval requires a progress callback")


def process_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(os.environ)
    if environment:
        result.update(environment)
    return result


def check_responses(
    responses: list[dict[str, Any]], check: bool
) -> list[dict[str, Any]]:
    if check and any(response.get("error") for response in responses):
        raise BatchExecutionError(responses)
    return responses


def parse_responses(
    stdout: str, stderr: str, expected: int, returncode: int
) -> list[dict[str, Any]]:
    if returncode != 0:
        raise RuntimeError(f"Go batch exited with {returncode}: {stderr.strip()}")
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(responses) != expected:
        raise RuntimeError(f"expected {expected} responses, received {len(responses)}")
    return responses


def _quiet_batch(
    binary: str,
    requests: list[Mapping[str, Any]],
    environment: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [binary, "batch"],
        input=payload(requests),
        text=True,
        capture_output=True,
        env=process_environment(environment),
        check=False,
    )
    return parse_responses(
        completed.stdout, completed.stderr, len(requests), completed.returncode
    )


def _streaming_batch(
    binary: str,
    requests: list[Mapping[str, Any]],
    environment: Mapping[str, str] | None,
    progress: ProgressCallback,
    progress_step_interval: int,
) -> list[dict[str, Any]]:
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
        env=process_environment(environment),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    callback_errors: list[Exception] = []

    def read_stdout() -> None:
        stdout_lines.extend(process.stdout)

    def read_stderr() -> None:
        for line in process.stderr:
            stderr_lines.append(line)
            if callback_errors:
                continue
            try:
                progress(json.loads(line))
            except json.JSONDecodeError:
                continue
            except Exception as error:  # noqa: BLE001
                callback_errors.append(error)

    stdout_thread = threading.Thread(target=read_stdout, name="smp-stdout")
    stderr_thread = threading.Thread(target=read_stderr, name="smp-progress")
    stdout_thread.start()
    stderr_thread.start()
    try:
        process.stdin.write(payload(requests))
        process.stdin.close()
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
        process.stdout.close()
        process.stderr.close()
    if callback_errors:
        raise callback_errors[0]
    return parse_responses(
        "".join(stdout_lines), "".join(stderr_lines), len(requests), returncode
    )


def run_batch_process(
    binary_path_value: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    *,
    check: bool,
    environment: Mapping[str, str] | None,
    progress: ProgressCallback | None,
    progress_step_interval: int,
) -> list[dict[str, Any]]:
    request_list = list(requests)
    if not request_list:
        return []
    validate_progress(progress, progress_step_interval)
    binary = binary_path(binary_path_value)
    if progress is None:
        responses = _quiet_batch(binary, request_list, environment)
    else:
        total = len(request_list)

        def enrich(event: dict[str, Any]) -> None:
            item = dict(event)
            item["request_index"] = int(item.get("batch_index", 0))
            item["request_total"] = total
            item["process_index"] = 1
            item["process_total"] = 1
            progress(item)

        responses = _streaming_batch(
            binary, request_list, environment, enrich, progress_step_interval
        )
    return check_responses(responses, check)


def run_parallel_processes(
    binary_path_value: os.PathLike[str] | str,
    requests: Iterable[Mapping[str, Any]],
    processes: int,
    *,
    check: bool,
    environment: Mapping[str, str] | None,
    progress: ProgressCallback | None,
    progress_step_interval: int,
) -> list[dict[str, Any]]:
    request_list = list(requests)
    if not request_list:
        return []
    if processes < 1 or processes > len(request_list):
        raise ValueError("processes must be in [1, number of requests]")
    validate_progress(progress, progress_step_interval)
    binary = binary_path(binary_path_value)
    pending: queue.Queue[tuple[int, Mapping[str, Any]]] = queue.Queue()
    for index, request in enumerate(request_list):
        pending.put((index, request))
    ordered: list[dict[str, Any] | None] = [None] * len(request_list)
    callback_lock = threading.Lock()
    stop = threading.Event()

    def worker(process_index: int) -> list[tuple[int, dict[str, Any]]]:
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
            env=process_environment(environment),
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        errors: list[Exception] = []
        stderr_lines: list[str] = []
        local_to_global: dict[int, int] = {}
        mapping_lock = threading.Lock()

        def read_progress() -> None:
            for line in process.stderr:
                stderr_lines.append(line)
                if progress is None or errors:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local_index = int(item.get("batch_index", 0))
                with mapping_lock:
                    global_index = local_to_global.get(local_index, -1)
                item["request_index"] = global_index + 1
                item["request_total"] = len(request_list)
                item["process_index"] = process_index + 1
                item["process_total"] = processes
                try:
                    with callback_lock:
                        progress(item)
                except Exception as error:  # noqa: BLE001
                    errors.append(error)

        stderr_thread = threading.Thread(
            target=read_progress, name=f"smp-progress-{process_index + 1}"
        )
        stderr_thread.start()
        completed: list[tuple[int, dict[str, Any]]] = []
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
                process.stdin.write(payload([request]))
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"Go batch ended before request {global_index + 1}: "
                        f"{''.join(stderr_lines).strip()}"
                    )
                completed.append((global_index, json.loads(line)))
            process.stdin.close()
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"Go batch exited with {returncode}: {''.join(stderr_lines).strip()}"
                )
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
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            stderr_thread.join()
            process.stdout.close()
            process.stderr.close()
        if errors:
            raise errors[0]
        return completed

    with ThreadPoolExecutor(max_workers=processes, thread_name_prefix="smp-go") as pool:
        futures = [pool.submit(worker, index) for index in range(processes)]
        for future in futures:
            for index, response in future.result():
                ordered[index] = response
    responses = [item for item in ordered if item is not None]
    if len(responses) != len(request_list):
        raise RuntimeError("parallel result assembly lost responses")
    return check_responses(responses, check)


def print_progress(event: Mapping[str, Any]) -> None:
    """Render progress events from either Go command."""

    request = event.get("request_id", "<unknown>")
    location = f"[{event.get('request_index', '?')}/{event.get('request_total', '?')}"
    if int(event.get("process_total", 1)) > 1:
        location += f" p{event['process_index']}"
    location += "]"
    kind = str(event.get("event", "progress"))
    if kind == "path_heartbeat":
        detail = (
            f"{event.get('stage')} path {event.get('path_index')}/"
            f"{event.get('total_paths')} step {event.get('step')}"
        )
    elif kind == "path_completed":
        detail = (
            f"{event.get('stage')} {event.get('completed_paths')}/"
            f"{event.get('total_paths')} paths; last={event.get('category')}"
        )
    elif kind == "step_heartbeat":
        detail = f"step {event.get('step')}/{event.get('total_steps')}"
    elif kind.startswith("scenario_"):
        detail = f"scenario {event.get('scenario_index')}/{event.get('scenario_count')}"
    elif kind in {"request_rejected", "request_failed"}:
        detail = f"{kind}: {event.get('message', '<no detail>')}"
    else:
        detail = kind
    print(
        f"{location} {request}: {detail} "
        f"({float(event.get('elapsed_seconds', 0.0)):.1f}s)",
        file=sys.stderr,
        flush=True,
    )
