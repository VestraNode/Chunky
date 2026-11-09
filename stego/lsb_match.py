# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbor_prediction(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = image.astype(np.float32)
    h, w, _ = f.shape
    total = np.zeros_like(f)
    count = np.zeros((h, w), dtype=np.float32)
    total[1:] += f[:-1]; count[1:] += 1
    total[:-1] += f[1:]; count[:-1] += 1
    total[:, 1:] += f[:, :-1]; count[:, 1:] += 1
    total[:, :-1] += f[:, 1:]; count[:, :-1] += 1
    return total / np.maximum(count, 1)[..., None], count


def _pred_at(image: np.ndarray, r: np.ndarray, c: np.ndarray, ch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = image.shape
    total = np.zeros(r.shape, dtype=np.float32)
    count = np.zeros(r.shape, dtype=np.float32)
    for dr, dc in _DIRECTIONS:
        qr, qc = r + dr, c + dc
        inside = (qr >= 0) & (qr < h) & (qc >= 0) & (qc < w)
        total += np.where(inside, image[np.clip(qr, 0, h - 1), np.clip(qc, 0, w - 1), ch].astype(np.float32), np.float32(0))
        count += inside
    return total / np.maximum(count, 1), count


def _neighborhood(image: np.ndarray, pred: Optional[np.ndarray], count: Optional[np.ndarray],
                  rr: np.ndarray, cc: np.ndarray, ch: np.ndarray):
    h, w, _ = image.shape
    p = _pred_at(image, rr, cc, ch)[0] if pred is None else pred[rr, cc, ch]
    sides = []
    for dr, dc in _DIRECTIONS:
        qr, qc = rr + dr, cc + dc
        inside = (qr >= 0) & (qr < h) & (qc >= 0) & (qc < w)
        qr, qc = np.clip(qr, 0, h - 1), np.clip(qc, 0, w - 1)
        vq = image[qr, qc, ch].astype(np.float32)
        pq, cq = _pred_at(image, qr, qc, ch) if pred is None else (pred[qr, qc, ch], count[qr, qc])
        sides.append((inside, vq, pq, np.maximum(cq, 1)))
    return p, sides


def _score_nb(image: np.ndarray, nb, rr: np.ndarray, cc: np.ndarray, ch: np.ndarray, delta: int,
              threshold: float, origin: Optional[np.ndarray] = None) -> np.ndarray:
    p, sides = nb
    v = image[rr, cc, ch].astype(np.float32)
    new = v + delta
    valid = (new >= 0) & (new <= 255)
    if origin is not None:
        valid &= np.abs(new - origin[rr, cc, ch].astype(np.float32)) <= 1

    dev_before, dev_after = np.abs(v - p), np.abs(new - p)
    flags = (dev_after > threshold).astype(np.float32) - (dev_before > threshold)
    dev_sum = dev_after - dev_before

    for inside, vq, pq, cq in sides:
        before = np.abs(vq - pq)
        after = np.abs(vq - (pq + delta / cq))
        flags += inside * ((after > threshold).astype(np.float32) - (before > threshold))
        dev_sum += inside * (after - before)

    return np.where(valid, 1000.0 * flags + dev_sum, np.inf)


def _score(image: np.ndarray, pred: Optional[np.ndarray], count: Optional[np.ndarray], rr: np.ndarray, cc: np.ndarray,
           ch: np.ndarray, delta: int, threshold: float, origin: Optional[np.ndarray] = None) -> np.ndarray:
    return _score_nb(image, _neighborhood(image, pred, count, rr, cc, ch), rr, cc, ch, delta, threshold, origin)


def best_deltas(image: np.ndarray, pred: Optional[np.ndarray], count: Optional[np.ndarray], rr: np.ndarray,
                cc: np.ndarray, ch: np.ndarray, threshold: float,
                origin: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    nb = _neighborhood(image, pred, count, rr, cc, ch)
    plus = _score_nb(image, nb, rr, cc, ch, +1, threshold, origin)
    minus = _score_nb(image, nb, rr, cc, ch, -1, threshold, origin)
    delta = np.where(plus <= minus, 1, -1)
    return delta, np.minimum(plus, minus)


def _prediction_for(image: np.ndarray, n_values: int):
    return neighbor_prediction(image) if n_values * 24 > image.size else (None, None)


def apply_lsb_matching(image: np.ndarray, rr: np.ndarray, cc: np.ndarray, ch: np.ndarray,
                       bits: np.ndarray, threshold: float = 8,
                       origin: Optional[np.ndarray] = None) -> int:
    rr, cc, ch, bits = (np.asarray(a) for a in (rr, cc, ch, bits))
    need = (image[rr, cc, ch] & 1) != bits
    rr, cc, ch = rr[need], cc[need], ch[need]
    changed = 0
    for color in (0, 1):
        sel = ((rr + cc) & 1) == color
        if not sel.any():
            continue
        r, c, k = rr[sel], cc[sel], ch[sel]
        pred, count = _prediction_for(image, r.size)
        delta, _ = best_deltas(image, pred, count, r, c, k, threshold, origin)
        image[r, c, k] = (image[r, c, k].astype(np.int16) + delta).astype(np.uint8)
        changed += int(r.size)
    return changed


def apply_low_bits(image: np.ndarray, rr: np.ndarray, cc: np.ndarray, ch: np.ndarray,
                   targets: np.ndarray, nbits: int) -> None:
    rr, cc, ch, targets = (np.asarray(a) for a in (rr, cc, ch, targets))
    mask = (1 << nbits) - 1
    value = image[rr, cc, ch].astype(np.int16)
    base = (value & ~mask) | targets.astype(np.int16)
    step = 1 << nbits
    options = np.stack([base, base + step, base - step])
    distance = np.where((options >= 0) & (options <= 255), np.abs(options - value), 10 ** 6)
    best = np.argmin(distance, axis=0)
    image[rr, cc, ch] = options[best, np.arange(value.size)].astype(np.uint8)


def toggle_costs(image: np.ndarray, rr: np.ndarray, cc: np.ndarray, ch: np.ndarray,
                 threshold: float = 8, origin: Optional[np.ndarray] = None) -> np.ndarray:
    rr, cc, ch = np.asarray(rr), np.asarray(cc), np.asarray(ch)
    pred, count = _prediction_for(image, rr.size)
    return best_deltas(image, pred, count, rr, cc, ch, threshold, origin)[1]
