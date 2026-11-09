# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

from typing import Tuple

import numpy as np

from .analysis import bit_count_matrix, steganalysis


def chunk_noise_grid(analysis_matrix: np.ndarray, chunk_size: int) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size має бути додатним")

    h, w = analysis_matrix.shape
    rows = h // chunk_size
    cols = w // chunk_size
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=np.int64)

    noise_per_pixel = bit_count_matrix(analysis_matrix).astype(np.int64)
    cropped = noise_per_pixel[: rows * chunk_size, : cols * chunk_size]
    reshaped = cropped.reshape(rows, chunk_size, cols, chunk_size)
    noise_grid = reshaped.sum(axis=(1, 3))
    return noise_grid


def rank_chunks(noise_grid: np.ndarray, descending: bool = True) -> np.ndarray:
    rows, cols = noise_grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((0, 2), dtype=np.int64)

    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    coords = np.stack([r_idx.ravel(), c_idx.ravel()], axis=1)
    scores = noise_grid.ravel()

    order = np.argsort(scores, kind="stable")
    if descending:
        order = order[::-1]
    return coords[order]


def count_chunks_above(noise_grid: np.ndarray, threshold: float) -> int:
    if noise_grid.size == 0:
        return 0
    return int(np.sum(noise_grid > threshold))


def chunk_rating(analysis_matrix: np.ndarray, chunk_size: int,
                  descending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    noise_grid = chunk_noise_grid(analysis_matrix, chunk_size)
    order = rank_chunks(noise_grid, descending=descending)
    return order, noise_grid


def count_chunks_with_noise_above(image: np.ndarray, chunk_size: int, threshold_x: float,
                                   analysis_threshold: int = 8) -> int:
    matrix = steganalysis(image, threshold=analysis_threshold)
    noise_grid = chunk_noise_grid(matrix, chunk_size)
    return count_chunks_above(noise_grid, threshold_x)
