# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import logging

import numpy as np

from ..analysis import steganalysis
from ..chunking import chunk_rating
from ..compression import METHOD_NAMES, compress_best
from ..common import (
    COMPRESSION_FIELD_BITS,
    FIRST_CHUNK_MARKER_PIXELS,
    HEADER_TOTAL_BITS,
    LENGTH_FIELD_BITS,
    MAGIC_BYTE,
    MAGIC_LEN_BITS,
    MARKER_BITS_PER_CHANNEL,
    MARKER_COUNT_FIELD_BITS,
    MAX_MARKER_PIXELS,
    Y_FIELD_BITS,
    bytes_to_bits,
    chunks_needed,
    data_pixels_per_chunk,
    data_positions_for_chunk,
    embed_bitstream,
    embed_marker,
    find_marker_cells,
    invalidate_marker_cache,
    int_to_bits,
    marker_pixels_needed,
    marker_value_map,
    ordered_chunk_positions,
    plan_marker_pixels,
)

_BITS_PER_MARKER_PIXEL = MARKER_BITS_PER_CHANNEL * 3
log = logging.getLogger("stego.encoder")


class _MarkerCollision(ValueError):
    pass


def capacity_bytes(num_chunks: int, x: int, y: int, use_key: bool) -> int:
    if num_chunks <= 0:
        return 0
    m = marker_pixels_needed(num_chunks, num_chunks)
    first = data_pixels_per_chunk(x, use_key, FIRST_CHUNK_MARKER_PIXELS)
    other = min(y, data_pixels_per_chunk(x, use_key, m))
    if first <= 0 or other < 0:
        return 0
    pixels_total = first + (num_chunks - 1) * other
    return max(0, (pixels_total * 3 - HEADER_TOTAL_BITS) // 8)


def _neutralize_false_markers(img: np.ndarray, x: int, grid_shape, used: dict,
                               marker_px_of: dict, m: int, use_key: bool) -> int:
    rows, cols = grid_shape
    last_number = len(used)
    bpc = MARKER_BITS_PER_CHANNEL

    own_marker_pixels = {
        (r * x + py, c * x + px)
        for (r, c), number in used.items() for (py, px) in marker_px_of[number]
    }

    fixes = 0
    cache: dict = {}
    for _ in range(10):
        changed = False
        vmap = None if use_key else marker_value_map(img, x, rows, cols, marker_px_of[2]) if 2 in marker_px_of else {}
        for number, true_cell in sorted((n, cell) for cell, n in used.items()):
            if number == 1:
                continue
            offsets = marker_px_of[number]
            flips = [
                (j, ch, b, _BITS_PER_MARKER_PIXEL * (m - 1 - j) + (2 - ch) * bpc + b)
                for b in range(bpc) for j in range(m) for ch in (2, 1, 0)
            ]
            if vmap is None:
                rr, cc = find_marker_cells(img, x, rows, cols, offsets, number, cache=cache)
                hits = list(zip(rr.tolist(), cc.tolist()))
            else:
                hits = [divmod(idx, cols) for idx in vmap.get(number, [])]
            for r, c in hits:
                if (r, c) == true_cell:
                    continue
                cell_used = (r, c) in used
                for j, ch, b, shift in flips:
                    new_value = number ^ (1 << shift)
                    if new_value != 0 and new_value <= last_number:
                        continue
                    py, px = r * x + offsets[j][0], c * x + offsets[j][1]
                    if cell_used and (b == 0 or (py, px) in own_marker_pixels):
                        continue
                    img[py, px, ch] ^= 1 << b
                    invalidate_marker_cache(cache, py, px, x)
                    changed = True
                    fixes += 1
                    break
                else:
                    raise _MarkerCollision(f"Збіг маркера чанка №{number} з іншим чанком.")
        if not changed:
            return fixes
    raise _MarkerCollision("Не вдалося усунути хибні збіги маркерів чанків.")


def encode(image: np.ndarray, x: int, y: int, data: bytes, use_key: bool = False,
           key: int = 0, threshold: int = 8, compress: bool = True,
           optimize_markers: bool = False) -> np.ndarray:
    return encode_with_report(image, x, y, data, use_key, key, threshold, compress,
                              optimize_markers)[0]


def encode_with_report(image: np.ndarray, x: int, y: int, data: bytes, use_key: bool = False,
                       key: int = 0, threshold: int = 8, compress: bool = True,
                       optimize_markers: bool = False):
    from .decoder import decode_with_report

    analysis_matrix = steganalysis(image, threshold=threshold)
    order, noise_grid = chunk_rating(analysis_matrix, x, descending=True)
    grid_chunks = int(order.shape[0])
    if grid_chunks == 0:
        raise ValueError("Зображення замале для чанків такого розміру (x).")

    if data_pixels_per_chunk(x, use_key, FIRST_CHUNK_MARKER_PIXELS) * 3 < HEADER_TOTAL_BITS:
        raise ValueError(
            f"Розмір чанка x={x} замалий: службовий заголовок ({HEADER_TOTAL_BITS} біт) "
            f"не вміщується у чанк №1. Збільште x.")

    original_bytes = len(data)
    if compress:
        method, stored, candidates = compress_best(data)
        if method:
            log.info("Архівація: %s, %d -> %d байт (%.1f%% від початкового)",
                     METHOD_NAMES[method], original_bytes, len(stored), 100 * len(stored) / max(1, original_bytes))
        else:
            log.warning("Архівація не дала виграшу (%s) — зберігаю без стиснення.",
                        ", ".join(f"{k}: {v}" for k, v in candidates.items()) or "дані порожні")
    else:
        method, stored, candidates = 0, data, {}

    payload_bits = bytes_to_bits(stored)
    total_bits = HEADER_TOTAL_BITS + len(payload_bits)
    m_safe, chunks_expected = plan_marker_pixels(grid_chunks, total_bits, x, y, use_key)
    if chunks_expected == 0:
        raise ValueError(
            f"Недостатньо місткості для приховування даних ({total_bits} біт разом із "
            f"заголовком). Збільште x, y, увімкніть архівацію або зменште обсяг даних.")
    log.info("Маркер: безпечне M=%d пікселів, потрібно ≈%d чанків із %d.", m_safe, chunks_expected, grid_chunks)

    m_start = FIRST_CHUNK_MARKER_PIXELS if optimize_markers else m_safe
    trials = []
    result = None
    for m in range(m_start, MAX_MARKER_PIXELS + 1):
        n_est = chunks_needed(total_bits, x, y, use_key, m)
        if n_est == 0 or n_est > grid_chunks:
            trials.append((m, "не вистачає місткості"))
            break
        if n_est >> (_BITS_PER_MARKER_PIXEL * m):
            trials.append((m, "номер чанка не влазить у маркер"))
            continue
        try:
            img, used, fixes = _encode_with_marker_pixels(
                image, order, noise_grid.shape, payload_bits, x, y, use_key, key, m, method)
        except _MarkerCollision:
            trials.append((m, "збіги маркерів не усунути"))
            if optimize_markers:
                log.warning("Оптимізація: M=%d — збіги маркерів не усунути, пробую M=%d.", m, m + 1)
            continue
        if optimize_markers:
            try:
                check, _ = decode_with_report(img, use_key, key, threshold, x_candidates=[x], quiet=True)
            except ValueError:
                check = None
            if check != data:
                trials.append((m, "не пройшло перевірку декодуванням"))
                log.warning("Оптимізація: M=%d — результат не декодується, пробую M=%d.", m, m + 1)
                continue
        result = (m, img, used, fixes)
        break

    if result is None:
        raise ValueError(
            "Не вдалося підібрати довжину маркера без збігів ("
            + "; ".join(f"M={m}: {why}" for m, why in trials) + "). Спробуйте інший ключ або x.")
    m, img, used, fixes = result
    if optimize_markers:
        log.info("Оптимізація: мінімальний робочий M=%d (безпечний був %d).", m, m_safe)
    if fixes:
        log.warning("Усунено %d випадкових збігів маркерів (змінено по 1 біту в сторонніх чанках).", fixes)

    marker_bits = (FIRST_CHUNK_MARKER_PIXELS + (len(used) - 1) * m) * _BITS_PER_MARKER_PIXEL
    report = {
        "chunk_size": x,
        "y": y,
        "use_key": use_key,
        "grid_rows": int(noise_grid.shape[0]),
        "grid_cols": int(noise_grid.shape[1]),
        "chunks_total": grid_chunks,
        "chunks_used": len(used),
        "used_cells": [(r, c, n) for (r, c), n in used.items()],
        "marker_pixels": m,
        "marker_pixels_safe": m_safe,
        "optimized": optimize_markers,
        "compression": METHOD_NAMES[method],
        "compression_method": method,
        "compression_candidates": candidates,
        "original_bytes": original_bytes,
        "stored_bytes": len(stored),
        "payload_bits": len(payload_bits),
        "header_bits": HEADER_TOTAL_BITS,
        "marker_bits": marker_bits,
        "used_bits_total": len(payload_bits) + HEADER_TOTAL_BITS + marker_bits,
        "capacity_bytes": capacity_bytes(grid_chunks, x, y, use_key),
        "marker_fixes": fixes,
    }
    log.info("Закодовано: %d чанків із %d, M=%d, збережено %d байт.", len(used), grid_chunks, m, len(stored))
    return img, report


def _encode_with_marker_pixels(image: np.ndarray, order: np.ndarray, grid_shape, payload_bits,
                                x: int, y: int, use_key: bool, key: int, m: int,
                                compression_method: int = 0):
    img = image.copy()
    header_bits = (
        int_to_bits(MAGIC_BYTE, MAGIC_LEN_BITS)
        + int_to_bits(len(payload_bits), LENGTH_FIELD_BITS)
        + int_to_bits(y, Y_FIELD_BITS)
        + int_to_bits(m, MARKER_COUNT_FIELD_BITS)
        + int_to_bits(compression_method, COMPRESSION_FIELD_BITS)
    )
    full_bits = header_bits + payload_bits
    total_needed = len(full_bits)

    cursor = 0
    chunk_number = 1
    used: dict = {}
    marker_px_of: dict = {}
    for (r, c) in order:
        if cursor >= total_needed:
            break

        marker_pixels = FIRST_CHUNK_MARKER_PIXELS if chunk_number == 1 else m
        top_left = (int(r) * x, int(c) * x)
        key_i = key + (chunk_number - 1) if use_key else 0

        ordered_positions = ordered_chunk_positions(top_left, x, use_key, key_i)
        if len(ordered_positions) <= marker_pixels:
            continue

        embed_marker(img, ordered_positions[:marker_pixels], chunk_number)

        data_positions = data_positions_for_chunk(top_left, x, use_key, key_i, marker_pixels)
        if chunk_number != 1:
            data_positions = data_positions[:y]

        remaining = full_bits[cursor: cursor + len(data_positions) * 3]
        cursor += embed_bitstream(img, data_positions, remaining)

        used[(int(r), int(c))] = chunk_number
        marker_px_of[chunk_number] = [
            (py - top_left[0], px - top_left[1]) for (py, px) in ordered_positions[:marker_pixels]
        ]
        chunk_number += 1

    if cursor < total_needed:
        raise ValueError(
            f"Недостатньо місткості для приховування даних: записано {cursor} "
            f"з {total_needed} біт. Збільште x, y або зменшіть обсяг даних."
        )

    fixes = _neutralize_false_markers(img, x, grid_shape, used, marker_px_of, m, use_key)
    return img, used, fixes
