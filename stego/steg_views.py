# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from .analysis import expand_flags, steganalysis

DEFAULT_STOPS = ("#440154", "#3b528b", "#21908d", "#5dc863", "#fde725")

VIEWS: Dict[str, dict] = {
    "neighbors": {
        "name": "Відхилення від сусідів",
        "pair": False, "gradient": False,
        "desc": "Світиться канал, що відхиляється від прогнозу за сусідами більше ніж на поріг: R — червоний, G — зелений, B — синій.",
    },
    "lsb": {
        "name": "LSB-площина (біт 0)",
        "pair": False, "gradient": False,
        "desc": "Молодший біт кожного каналу (R, G, B → червоний, зелений, синій). Вбудовані дані виглядають як рівномірний шум там, де фото було гладким.",
    },
    "bit1": {
        "name": "Бітова площина 1",
        "pair": False, "gradient": False,
        "desc": "Другий молодший біт кожного каналу. Його змінюють лише маркери чанків (вони пишуться в 3 молодші біти).",
    },
    "bit2": {
        "name": "Бітова площина 2",
        "pair": False, "gradient": False,
        "desc": "Третій молодший біт кожного каналу. Його змінюють лише маркери чанків.",
    },
    "lsb_entropy": {
        "name": "Ентропія LSB-площини",
        "pair": False, "gradient": True,
        "desc": "Ентропія пар сусідніх молодших бітів у блоках 8×8: що яскравіше, то випадковіше (схоже на вбудовані дані). 0 — повністю впорядковано.",
    },
    "chi2": {
        "name": "Хі-квадрат атака",
        "pair": False, "gradient": True,
        "desc": "Атака Вестфельда: у блоках 16×16 частоти значень 2k і 2k+1 вирівнюються після LSB-вбудовування. Яскравіше — вища ймовірність вбудовування.",
    },
    "residual": {
        "name": "Високочастотний залишок",
        "pair": False, "gradient": True,
        "desc": "Модуль лапласіана (піксель мінус середнє сусідів), підсилений: показує, де на фото є дрібний шум і текстура.",
    },
    "diff": {
        "name": "Карта змін (після − до)",
        "pair": True, "gradient": True,
        "desc": "Лише для пари фото: чорне — без змін, колір — на скільки (1…8 і більше) змінився найбільш змінений канал пікселя.",
    },
}


def _lut(stops: Sequence[str]) -> np.ndarray:
    pts = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=np.float64)
    xs = np.linspace(0, 1, len(pts))
    grid = np.linspace(0, 1, 256)
    return np.stack([np.interp(grid, xs, pts[:, c]) for c in range(3)], axis=1).astype(np.uint8)


gradient_lut = _lut


def _colorize(values01: np.ndarray, stops: Sequence[str]) -> np.ndarray:
    idx = np.clip(np.rint(values01 * 255), 0, 255).astype(np.int64)
    return _lut(stops)[idx]


def _upsample(block_map: np.ndarray, block: int, height: int, width: int) -> np.ndarray:
    big = np.repeat(np.repeat(block_map, block, axis=0), block, axis=1)
    pad_h, pad_w = height - big.shape[0], width - big.shape[1]
    if pad_h > 0 or pad_w > 0:
        big = np.pad(big, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
    return big[:height, :width]


def _erfc(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-ax * ax)
    return 1.0 - sign * erf


def _chi2_survival(chi: np.ndarray, df: np.ndarray) -> np.ndarray:
    df = np.maximum(df, 1).astype(np.float64)
    z = ((chi / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / np.sqrt(2.0 / (9.0 * df))
    return 0.5 * _erfc(z / math.sqrt(2.0))


def _bitplane(image: np.ndarray, bit: int) -> np.ndarray:
    return (((image >> bit) & 1) * 255).astype(np.uint8)


def _lsb_entropy(image: np.ndarray, block: int = 8) -> np.ndarray:
    h, w, _ = image.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return np.zeros((h, w))
    lsb = (image[: hb * block, : wb * block] & 1).astype(np.int64)
    code = lsb[:, :-1, :] * 2 + lsb[:, 1:, :]
    code = np.concatenate([code, code[:, -1:, :]], axis=1)
    blk = (np.arange(hb * block) // block)[:, None] * wb + (np.arange(wb * block) // block)[None, :]
    entropy = np.zeros(hb * wb)
    for c in range(3):
        counts = np.bincount((blk * 4 + code[:, :, c]).ravel(), minlength=hb * wb * 4)
        p = counts.reshape(hb * wb, 4) / counts.reshape(hb * wb, 4).sum(axis=1, keepdims=True)
        entropy += -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1) / 2.0 / 3.0
    return _upsample(entropy.reshape(hb, wb), block, h, w)


def _chi2_map(image: np.ndarray, block: int = 16) -> np.ndarray:
    h, w, _ = image.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return np.zeros((h, w))
    blk = (np.arange(hb * block) // block)[:, None] * wb + (np.arange(wb * block) // block)[None, :]
    prob = np.zeros(hb * wb)
    for c in range(3):
        values = image[: hb * block, : wb * block, c].astype(np.int64)
        counts = np.bincount((blk * 256 + values).ravel(), minlength=hb * wb * 256).reshape(hb * wb, 256)
        even, odd = counts[:, 0::2].astype(np.float64), counts[:, 1::2].astype(np.float64)
        total = even + odd
        valid = total >= 6
        expected = np.where(valid, total / 2.0, 1.0)
        chi = np.where(valid, (even - expected) ** 2 / expected, 0.0).sum(axis=1)
        df = valid.sum(axis=1) - 1
        p = np.where(df >= 4, _chi2_survival(chi, df), 0.0)
        prob += p / 3.0
    return _upsample(prob.reshape(hb, wb), block, h, w)


def _residual(image: np.ndarray) -> np.ndarray:
    f = image.astype(np.int32)
    p = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lap = 4 * f - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]
    return np.clip(np.abs(lap).max(axis=2) / 24.0, 0.0, 1.0)


def render_view(view: str, image: np.ndarray, threshold: int = 8,
                original: Optional[np.ndarray] = None,
                stops: Optional[Sequence[str]] = None) -> np.ndarray:
    if view not in VIEWS:
        raise ValueError(f"Невідомий вигляд стегоаналізу: {view}")
    stops = stops or DEFAULT_STOPS

    if view == "neighbors":
        return expand_flags(steganalysis(image, threshold)).astype(np.uint8) * 255
    if view == "lsb":
        return _bitplane(image, 0)
    if view == "bit1":
        return _bitplane(image, 1)
    if view == "bit2":
        return _bitplane(image, 2)
    if view == "lsb_entropy":
        return _colorize(_lsb_entropy(image), stops)
    if view == "chi2":
        return _colorize(_chi2_map(image), stops)
    if view == "residual":
        return _colorize(_residual(image), stops)

    if original is None or original.shape != image.shape:
        raise ValueError("Для карти змін потрібне оригінальне фото того самого розміру.")
    magnitude = np.abs(original.astype(np.int32) - image.astype(np.int32)).max(axis=2)
    colored = _colorize(np.clip(magnitude / 8.0, 0.15, 1.0), stops)
    colored[magnitude == 0] = 0
    return colored


def _flag_palette() -> np.ndarray:
    i = np.arange(8)
    return np.stack([(i & 1) * 255, ((i >> 1) & 1) * 255, ((i >> 2) & 1) * 255], axis=1).astype(np.uint8)


def _color_index(values01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values01 * 255), 0, 255).astype(np.uint8)


def _bitplane_index(image: np.ndarray, bit: int) -> np.ndarray:
    plane = (image >> bit) & 1
    return plane[..., 0] | (plane[..., 1] << 1) | (plane[..., 2] << 2)


def render_view_indexed(view: str, image: np.ndarray, threshold: int = 8,
                        original: Optional[np.ndarray] = None,
                        stops: Optional[Sequence[str]] = None):
    if view not in VIEWS:
        raise ValueError(f"Невідомий вигляд стегоаналізу: {view}")
    stops = stops or DEFAULT_STOPS

    if view == "neighbors":
        return steganalysis(image, threshold), _flag_palette()
    if view in ("lsb", "bit1", "bit2"):
        return _bitplane_index(image, {"lsb": 0, "bit1": 1, "bit2": 2}[view]), _flag_palette()
    if view == "lsb_entropy":
        return _color_index(_lsb_entropy(image)), _lut(stops)
    if view == "chi2":
        return _color_index(_chi2_map(image)), _lut(stops)
    if view == "residual":
        return _color_index(_residual(image)), _lut(stops)

    if original is None or original.shape != image.shape:
        raise ValueError("Для карти змін потрібне оригінальне фото того самого розміру.")
    magnitude = np.abs(original.astype(np.int32) - image.astype(np.int32)).max(axis=2)
    index = _color_index(np.clip(magnitude / 8.0, 0.15, 1.0))
    index[magnitude == 0] = 0
    palette = _lut(stops).copy()
    palette[0] = 0
    return index, palette
