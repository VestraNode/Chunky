# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import bz2
import lzma
import zlib
from typing import Dict, Tuple

METHOD_NONE = 0
METHOD_NAMES: Dict[int, str] = {0: "без стиснення", 1: "zlib", 2: "bz2", 3: "lzma"}
LZMA_MAX_INPUT = 8 * 1024 * 1024


def _compress(method: int, data: bytes) -> bytes:
    if method == 1:
        return zlib.compress(data, 9)
    if method == 2:
        return bz2.compress(data, 9)
    if method == 3:
        return lzma.compress(data, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    raise ValueError(f"Невідомий метод стиснення: {method}")


def compress_best(data: bytes) -> Tuple[int, bytes, Dict[str, int]]:
    sizes: Dict[str, int] = {}
    best_method, best = METHOD_NONE, data
    if not data:
        return best_method, best, sizes
    for method in (1, 2, 3):
        if method == 3 and len(data) > LZMA_MAX_INPUT:
            continue
        packed = _compress(method, data)
        sizes[METHOD_NAMES[method]] = len(packed)
        if len(packed) < len(best):
            best_method, best = method, packed
    return best_method, best, sizes


def decompress(method: int, blob: bytes) -> bytes:
    if method == METHOD_NONE:
        return blob
    try:
        if method == 1:
            return zlib.decompress(blob)
        if method == 2:
            return bz2.decompress(blob)
        if method == 3:
            return lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    except Exception as exc:
        raise ValueError(f"Не вдалося розпакувати ({METHOD_NAMES.get(method, method)}): {exc}")
    raise ValueError(f"Невідомий метод стиснення: {method}")
