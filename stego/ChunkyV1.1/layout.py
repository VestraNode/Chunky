# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from ..common import _pixel_marker_values, key_shuffle_order

Position = Tuple[int, int]


@lru_cache(maxsize=300_000)
def _shuffled(x: int, key_i: int) -> Tuple[int, ...]:
    return tuple(key_shuffle_order(x * x, key_i))


def ordered_positions(top_left: Position, x: int, use_key: bool, key_i: int) -> List[Position]:
    r0, c0 = top_left
    indices = _shuffled(x, key_i) if use_key else range(x * x)
    return [(r0 + i // x, c0 + i % x) for i in indices]


def data_positions(top_left: Position, x: int, use_key: bool, key_i: int, marker_pixels: int) -> List[Position]:
    ordered = ordered_positions(top_left, x, use_key, key_i)
    return ordered[marker_pixels:] if use_key else ordered[marker_pixels::2]


@lru_cache(maxsize=300_000)
def marker_offsets(x: int, use_key: bool, key_i: int, marker_pixels: int) -> Tuple[Position, ...]:
    return tuple(ordered_positions((0, 0), x, use_key, key_i)[:marker_pixels])


class PixelGrids:

    def __init__(self, image: np.ndarray, x: int, rows: int, cols: int, bpc: int = 1):
        self.image, self.x, self.rows, self.cols, self.bpc = image, x, rows, cols, bpc
        self._cache: dict = {}
        self._index: dict = {}

    def grid(self, offset: Position) -> np.ndarray:
        g = self._cache.get(offset)
        if g is None:
            g = _pixel_marker_values(self.image, self.x, self.rows, self.cols, offset, self.bpc).astype(np.uint16)
            self._cache[offset] = g
        return g

    def _by_value(self, offset: Position):
        hit = self._index.get(offset)
        if hit is None:
            flat = self.grid(offset).ravel()
            order = np.argsort(flat, kind="stable")
            counts = np.bincount(flat, minlength=(1 << (3 * self.bpc)) + 1)
            hit = (order, np.concatenate(([0], np.cumsum(counts))))
            self._index[offset] = hit
        return hit

    def find(self, offsets, value: int):
        m = len(offsets)
        bits = 3 * self.bpc
        empty = np.zeros(0, dtype=np.int64)
        if value >> (bits * m):
            return empty, empty
        parts = [(value >> (bits * (m - 1 - j))) & ((1 << bits) - 1) for j in range(m)]
        order, starts = self._by_value(offsets[0])
        flat = order[starts[parts[0]]: starts[parts[0] + 1]]
        rr, cc = flat // self.cols, flat % self.cols
        for j in range(1, m):
            if rr.size == 0:
                break
            keep = self.grid(offsets[j])[rr, cc] == parts[j]
            rr, cc = rr[keep], cc[keep]
        return rr, cc
