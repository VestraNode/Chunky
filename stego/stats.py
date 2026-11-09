# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

from typing import Optional

import numpy as np

from .analysis import bit_count_matrix


def noise_channel_count(analysis_matrix: np.ndarray) -> int:
    return int(bit_count_matrix(analysis_matrix).sum())


def pollution_index(height: int, width: int, noise_channels: int) -> Optional[float]:
    if noise_channels <= 0:
        return None
    return (height * width * 3) / noise_channels


def diff_stats(original: np.ndarray, modified: np.ndarray) -> dict:
    diff = original.astype(np.int16) - modified.astype(np.int16)
    changed = diff != 0
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    return {
        "changed_values": int(changed.sum()),
        "changed_pixels": int(changed.any(axis=2).sum()),
        "total_values": int(changed.size),
        "max_abs_change": int(np.abs(diff).max()) if diff.size else 0,
        "mse": mse,
        "psnr_db": None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)),
    }


def noise_histogram(noise_grid: np.ndarray, max_possible: int) -> np.ndarray:
    return np.bincount(noise_grid.ravel(), minlength=max_possible + 1)[: max_possible + 1]
