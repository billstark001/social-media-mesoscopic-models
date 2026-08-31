"""Build helpers for local and editable installations."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

Backend = Literal["purego", "openblas", "accelerate"]


def _source_root(source_root: os.PathLike[str] | str | None) -> Path:
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent.parent
    if not (root / "go.mod").is_file():
        raise FileNotFoundError(
            f"cannot find go.mod under {root}; pass source_root explicitly"
        )
    return root


def _openblas_environment(environment: dict[str, str]) -> dict[str, str]:
    result = dict(environment)
    if shutil.which("brew"):
        completed = subprocess.run(
            ["brew", "--prefix", "openblas"],
            check=True,
            text=True,
            capture_output=True,
        )
        prefix = Path(completed.stdout.strip())
        library = prefix / "lib"
        include = prefix / "include"
        if not library.is_dir() or not include.is_dir():
            raise FileNotFoundError(
                f"OpenBLAS is not installed at the Homebrew prefix {prefix}"
            )
        result["CGO_CFLAGS"] = (result.get("CGO_CFLAGS", "") + f" -I{include}").strip()
        result["CGO_LDFLAGS"] = (
            result.get("CGO_LDFLAGS", "") + f" -L{library} -lopenblas"
        ).strip()
    elif not result.get("CGO_LDFLAGS"):
        raise RuntimeError(
            "OpenBLAS build requires Homebrew or explicit CGO_CFLAGS/CGO_LDFLAGS"
        )
    result["CGO_ENABLED"] = "1"
    return result


def build_binary(
    *,
    source_root: os.PathLike[str] | str | None = None,
    output_path: os.PathLike[str] | str | None = None,
    backend: Backend = "purego",
) -> Path:
    """Build ``cmd/smp-meso`` and return the absolute binary path.

    Model parameters have no defaults in the bindings. ``backend`` is only a
    build choice: pure Go is dependency-free, OpenBLAS is opt-in through CGO,
    and Accelerate is opt-in on Darwin.
    """

    root = _source_root(source_root)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "bin" / "smp-meso"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    command = ["go", "build", "-o", str(output)]
    if backend == "openblas":
        environment = _openblas_environment(environment)
        command.extend(["-tags", "openblas"])
    elif backend == "accelerate":
        if os.uname().sysname != "Darwin":
            raise RuntimeError("Accelerate is available only on Darwin")
        environment["CGO_ENABLED"] = "1"
        command.extend(["-tags", "accelerate"])
    elif backend != "purego":
        raise ValueError(f"unsupported backend {backend!r}")
    command.append("./cmd/smp-meso")
    subprocess.run(command, cwd=root, env=environment, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--backend", choices=("purego", "openblas", "accelerate"), required=True
    )
    arguments = parser.parse_args()
    path = build_binary(
        source_root=arguments.source_root,
        output_path=arguments.output,
        backend=arguments.backend,
    )
    print(path)


if __name__ == "__main__":
    main()
