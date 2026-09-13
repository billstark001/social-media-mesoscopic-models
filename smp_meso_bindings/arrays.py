"""Shared compressed float64 wire arrays for mesoscopic runtimes."""

from __future__ import annotations

import base64
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
        raise ValueError(
            f"encoded array contains {result.size} values, expected {expected}"
        )
    return result.reshape(shape).copy()
