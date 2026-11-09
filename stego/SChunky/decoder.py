# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import numpy as np

from ..analysis import steganalysis
from ..chunking import chunk_rating
from ..compression import METHOD_NAMES, decompress
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
    bits_to_bytes,
    bits_to_int,
    data_positions_for_chunk,
    extract_bitstream,
    find_marker_cells,
    marker_value_map,
    ordered_chunk_positions,
)

log = logging.getLogger("stego.decoder")

DEFAULT_X_CANDIDATES = range(4, 65)
MAX_FIRST_CHUNK_CANDIDATES = 200


def _find_chunk(image: np.ndarray, x: int, rows: int, cols: int, rank: np.ndarray, number: int,
                marker_pixels: int, use_key: bool, key: int, taken: set,
                value_map: Optional[dict] = None, cache: Optional[dict] = None) -> Optional[tuple]:
    if value_map is not None:
        matches = [divmod(idx, cols) for idx in value_map.get(number, [])]
    else:
        key_i = key + (number - 1) if use_key else 0
        offsets = ordered_chunk_positions((0, 0), x, use_key, key_i)[:marker_pixels]
        rr, cc = find_marker_cells(image, x, rows, cols, offsets, number, cache=cache)
        matches = list(zip(rr.tolist(), cc.tolist()))
    best, best_rank = None, None
    for r, c in matches:
        if (r, c) in taken:
            continue
        cell_rank = rank[r * cols + c]
        if best is None or cell_rank < best_rank:
            best, best_rank = (r, c), cell_rank
    return best


def _decode_from_first_chunk(image: np.ndarray, x: int, rows: int, cols: int, rank: np.ndarray,
                             first_cell: tuple, grid_chunks: int, use_key: bool, key: int, cache: Optional[dict] = None):
    top_left1 = (first_cell[0] * x, first_cell[1] * x)
    key_1 = key if use_key else 0
    positions1 = data_positions_for_chunk(top_left1, x, use_key, key_1, FIRST_CHUNK_MARKER_PIXELS)
    if len(positions1) * 3 < HEADER_TOTAL_BITS:
        return None

    stream1 = extract_bitstream(image, positions1, len(positions1) * 3)
    if bits_to_int(stream1[:MAGIC_LEN_BITS]) != MAGIC_BYTE:
        return None
    pos = MAGIC_LEN_BITS
    total_needed = bits_to_int(stream1[pos: pos + LENGTH_FIELD_BITS]); pos += LENGTH_FIELD_BITS
    y_value = bits_to_int(stream1[pos: pos + Y_FIELD_BITS]); pos += Y_FIELD_BITS
    marker_pixels = bits_to_int(stream1[pos: pos + MARKER_COUNT_FIELD_BITS]); pos += MARKER_COUNT_FIELD_BITS
    compression_method = bits_to_int(stream1[pos: pos + COMPRESSION_FIELD_BITS]); pos += COMPRESSION_FIELD_BITS

    if not FIRST_CHUNK_MARKER_PIXELS <= marker_pixels <= MAX_MARKER_PIXELS or y_value < 1:
        return None
    if compression_method not in METHOD_NAMES or total_needed > grid_chunks * x * x * 3:
        return None

    collected: List[int] = list(stream1[HEADER_TOTAL_BITS: HEADER_TOTAL_BITS + total_needed])
    used_cells = [(first_cell[0], first_cell[1], 1)]
    taken = {first_cell}

    value_map = None
    if not use_key and len(collected) < total_needed:
        offsets = ordered_chunk_positions((0, 0), x, False, 0)[:marker_pixels]
        value_map = marker_value_map(image, x, rows, cols, offsets)

    number = 2
    while len(collected) < total_needed:
        cell = _find_chunk(image, x, rows, cols, rank, number, marker_pixels, use_key, key, taken, value_map, cache)
        if cell is None:
            return None
        taken.add(cell)
        used_cells.append((cell[0], cell[1], number))

        key_i = key + (number - 1) if use_key else 0
        positions = data_positions_for_chunk(
            (cell[0] * x, cell[1] * x), x, use_key, key_i, marker_pixels)[:y_value]
        if not positions:
            return None
        need = total_needed - len(collected)
        bits = extract_bitstream(image, positions, min(need, len(positions) * 3))
        if not bits:
            return None
        collected.extend(bits)
        number += 1

    bits_per_pixel = MARKER_BITS_PER_CHANNEL * 3
    marker_bits = (FIRST_CHUNK_MARKER_PIXELS + (len(used_cells) - 1) * marker_pixels) * bits_per_pixel
    report = {
        "chunk_size": x,
        "y": y_value,
        "use_key": use_key,
        "grid_rows": rows,
        "grid_cols": cols,
        "chunks_total": grid_chunks,
        "chunks_used": len(used_cells),
        "used_cells": used_cells,
        "marker_pixels": marker_pixels,
        "payload_bits": total_needed,
        "header_bits": HEADER_TOTAL_BITS,
        "marker_bits": marker_bits,
        "used_bits_total": total_needed + HEADER_TOTAL_BITS + marker_bits,
    }
    stored = bits_to_bytes(collected[:total_needed])
    try:
        data = decompress(compression_method, stored)
    except ValueError:
        return None
    report["compression"] = METHOD_NAMES[compression_method]
    report["compression_method"] = compression_method
    report["stored_bytes"] = len(stored)
    report["original_bytes"] = len(data)
    return data, report


def _try_decode_with_x(image: np.ndarray, analysis_matrix: np.ndarray, x: int,
                       use_key: bool, key: int):
    order, noise_grid = chunk_rating(analysis_matrix, x, descending=True)
    grid_chunks = int(order.shape[0])
    if grid_chunks == 0:
        return None
    rows, cols = noise_grid.shape

    rank = np.empty(rows * cols, dtype=np.int64)
    rank[order[:, 0] * cols + order[:, 1]] = np.arange(grid_chunks)

    key_1 = key if use_key else 0
    offsets1 = ordered_chunk_positions((0, 0), x, use_key, key_1)[:FIRST_CHUNK_MARKER_PIXELS]
    cache: dict = {}
    rr, cc = find_marker_cells(image, x, rows, cols, offsets1, 1, cache=cache)
    if rr.size == 0:
        return None
    by_rank = np.argsort(rank[rr * cols + cc])[:MAX_FIRST_CHUNK_CANDIDATES]

    for idx in by_rank:
        result = _decode_from_first_chunk(
            image, x, rows, cols, rank, (int(rr[idx]), int(cc[idx])), grid_chunks, use_key, key, cache)
        if result is not None:
            return result
    return None


def decode(image: np.ndarray, use_key: bool = False, key: int = 0, threshold: int = 8,
           x_candidates: Optional[Iterable[int]] = None) -> bytes:
    return decode_with_report(image, use_key, key, threshold, x_candidates)[0]


def decode_with_report(image: np.ndarray, use_key: bool = False, key: int = 0,
                       threshold: int = 8, x_candidates: Optional[Iterable[int]] = None,
                       quiet: bool = False):
    candidates = x_candidates if x_candidates is not None else DEFAULT_X_CANDIDATES
    analysis_matrix = steganalysis(image, threshold=threshold)
    for x in candidates:
        try:
            result = _try_decode_with_x(image, analysis_matrix, x, use_key, key)
        except Exception:
            result = None
        if result is not None:
            rep = result[1]
            if not quiet:
                log.info("Декодовано: x=%d, M=%d, %d чанків, %d байт (%s).", rep["chunk_size"],
                         rep["marker_pixels"], rep["chunks_used"], rep["original_bytes"], rep["compression"])
            return result

    raise ValueError(
        "Не вдалося декодувати: жоден розмір чанка x зі списку кандидатів не підійшов "
        "(перевірте key/use_key, або розширте x_candidates)."
    )
