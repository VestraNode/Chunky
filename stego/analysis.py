# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULT_THRESHOLD = 8

PARALLEL_MIN_PIXELS = 1_000_000
BLOCK_ROWS = 128
_POOL = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)), thread_name_prefix="stego")
    return _POOL


_POPCOUNT = np.array([bin(v).count("1") for v in range(8)], dtype=np.uint8)


def _steganalysis_reference(image: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    img = image.astype(np.int32)
    h, w, _ = img.shape
    neighbor_sum = np.zeros((h, w, 3), dtype=np.int32)
    neighbor_cnt = np.zeros((h, w, 1), dtype=np.int32)
    neighbor_sum[1:, :, :] += img[:-1, :, :]
    neighbor_cnt[1:, :, 0] += 1
    neighbor_sum[:-1, :, :] += img[1:, :, :]
    neighbor_cnt[:-1, :, 0] += 1
    neighbor_sum[:, 1:, :] += img[:, :-1, :]
    neighbor_cnt[:, 1:, 0] += 1
    neighbor_sum[:, :-1, :] += img[:, 1:, :]
    neighbor_cnt[:, :-1, 0] += 1
    predicted = neighbor_sum / np.maximum(neighbor_cnt, 1)
    flags = np.abs(img - predicted) > threshold
    return (flags[:, :, 0].astype(np.uint8) | (flags[:, :, 1].astype(np.uint8) << 1) | (flags[:, :, 2].astype(np.uint8) << 2))


def _analysis_rows(image: np.ndarray, out: np.ndarray, r0: int, r1: int, threshold: int, horiz: np.ndarray) -> None:
    h = image.shape[0]
    a0, a1 = max(0, r0 - 1), min(h, r1 + 1)
    sub = image[a0:a1].astype(np.int16)
    neighbor_sum = np.zeros_like(sub)
    neighbor_sum[1:] += sub[:-1]
    neighbor_sum[:-1] += sub[1:]
    neighbor_sum[:, 1:] += sub[:, :-1]
    neighbor_sum[:, :-1] += sub[:, 1:]
    rows = np.arange(a0, a1)
    vert = (rows > 0).astype(np.int16) + (rows < h - 1).astype(np.int16)
    cnt = np.maximum(vert[:, None] + horiz[None, :], 1)[..., None]
    flags = np.abs(sub * cnt - neighbor_sum) > (threshold * cnt)
    packed = flags[..., 0].view(np.uint8) | (flags[..., 1].view(np.uint8) << 1) | (flags[..., 2].view(np.uint8) << 2)
    out[r0:r1] = packed[r0 - a0: r0 - a0 + (r1 - r0)]


def steganalysis(image: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Очікується зображення форми (H, W, 3) у RGB")
    if not (0 <= threshold < 255):
        return _steganalysis_reference(image, threshold)

    h, w, _ = image.shape
    out = np.empty((h, w), dtype=np.uint8)
    horiz = np.full(w, 2, dtype=np.int16)
    horiz[0] -= 1
    horiz[-1] -= 1
    if h * w < PARALLEL_MIN_PIXELS or h <= BLOCK_ROWS:
        _analysis_rows(image, out, 0, h, threshold, horiz)
        return out
    pool = _pool()
    list(pool.map(lambda r0: _analysis_rows(image, out, r0, min(h, r0 + BLOCK_ROWS), threshold, horiz), range(0, h, BLOCK_ROWS)))
    return out


def expand_flags(analysis_matrix: np.ndarray) -> np.ndarray:
    r = (analysis_matrix & 1).astype(bool)
    g = ((analysis_matrix >> 1) & 1).astype(bool)
    b = ((analysis_matrix >> 2) & 1).astype(bool)
    return np.stack([r, g, b], axis=-1)


def bit_count_matrix(analysis_matrix: np.ndarray) -> np.ndarray:
    return _POPCOUNT[analysis_matrix]
