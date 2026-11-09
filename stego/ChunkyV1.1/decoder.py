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
from ..common import (
    HEADER_TOTAL_BITS,
    LENGTH_FIELD_BITS,
    MAGIC_BYTE,
    MAGIC_LEN_BITS,
    MARKER_COUNT_FIELD_BITS,
    Y_FIELD_BITS,
    _pixel_marker_values,
    bits_to_bytes,
    bits_to_int,
    extract_bitstream,
)
from ..compression import METHOD_NAMES, decompress
from .layout import PixelGrids, data_positions, marker_offsets
from .params import (
    BITS_PER_CHANNEL,
    FIRST_CHUNK_MARKER_PIXELS,
    MAX_MARKER_BPC,
    MAX_MARKER_PIXELS,
    marker_value,
)

log = logging.getLogger("stego.chunky.decoder")

DEFAULT_X_CANDIDATES = range(4, 65)
MAX_FIRST_CHUNK_CANDIDATES = 300
BPC_FIELD_BITS = 2
METHOD_FIELD_BITS = 2


def ranked_value_map(image: np.ndarray, x: int, rows: int, cols: int, offsets, rank: np.ndarray,
                     bpc: int = BITS_PER_CHANNEL) -> dict:
    values = np.zeros((rows, cols), dtype=np.int64)
    for off in offsets:
        values = (values << (3 * bpc)) | _pixel_marker_values(image, x, rows, cols, off, bpc)
    flat = values.ravel()
    order = np.lexsort((rank[: flat.size], flat))
    sorted_values = flat[order]
    unique, start = np.unique(sorted_values, return_index=True)
    end = np.append(start[1:], sorted_values.size)
    return {int(u): order[s:e] for u, s, e in zip(unique, start, end)}


def candidates_for(grids: PixelGrids, x: int, cols: int, rank: np.ndarray, number: int,
                   marker_pixels: int, use_key: bool, key: int, value_map: Optional[dict]) -> np.ndarray:
    value = marker_value(number, marker_pixels, grids.bpc)
    if value_map is not None:
        return value_map.get(value, np.zeros(0, dtype=np.int64))
    key_i = key + (number - 1) if use_key else 0
    rr, cc = grids.find(marker_offsets(x, use_key, key_i, marker_pixels), value)
    flat = rr * cols + cc
    return flat[np.argsort(rank[flat], kind="stable")]


def _decode_from_first_chunk(image: np.ndarray, grids1: PixelGrids, x: int, rows: int, cols: int,
                             rank: np.ndarray, first_cell: tuple, grid_chunks: int, use_key: bool, key: int):
    top_left1 = (first_cell[0] * x, first_cell[1] * x)
    key_1 = key if use_key else 0
    positions1 = data_positions(top_left1, x, use_key, key_1, FIRST_CHUNK_MARKER_PIXELS)
    if len(positions1) * 3 < HEADER_TOTAL_BITS:
        return None

    stream1 = extract_bitstream(image, positions1, len(positions1) * 3)
    if bits_to_int(stream1[:MAGIC_LEN_BITS]) != MAGIC_BYTE:
        return None
    pos = MAGIC_LEN_BITS
    total_needed = bits_to_int(stream1[pos: pos + LENGTH_FIELD_BITS]); pos += LENGTH_FIELD_BITS
    y_value = bits_to_int(stream1[pos: pos + Y_FIELD_BITS]); pos += Y_FIELD_BITS
    marker_pixels = bits_to_int(stream1[pos: pos + MARKER_COUNT_FIELD_BITS]); pos += MARKER_COUNT_FIELD_BITS
    bpc = bits_to_int(stream1[pos: pos + BPC_FIELD_BITS]) + 1; pos += BPC_FIELD_BITS
    compression_method = bits_to_int(stream1[pos: pos + METHOD_FIELD_BITS])

    if not 1 <= marker_pixels <= MAX_MARKER_PIXELS or not 1 <= bpc <= MAX_MARKER_BPC or y_value < 1:
        return None
    if compression_method not in METHOD_NAMES or total_needed > grid_chunks * x * x * 3:
        return None

    grids = grids1 if bpc == grids1.bpc else PixelGrids(image, x, rows, cols, bpc)
    collected: List[int] = list(stream1[HEADER_TOTAL_BITS: HEADER_TOTAL_BITS + total_needed])
    used_cells = [(first_cell[0], first_cell[1], 1)]
    taken = np.zeros(rows * cols, dtype=bool)
    taken[first_cell[0] * cols + first_cell[1]] = True

    value_map = None
    if not use_key and len(collected) < total_needed:
        value_map = ranked_value_map(image, x, rows, cols, marker_offsets(x, False, 0, marker_pixels), rank, bpc)

    number = 2
    while len(collected) < total_needed:
        cand = candidates_for(grids, x, cols, rank, number, marker_pixels, use_key, key, value_map)
        chosen = next((int(i) for i in cand if not taken[i]), None)
        if chosen is None:
            return None
        taken[chosen] = True
        cell = divmod(chosen, cols)
        used_cells.append((cell[0], cell[1], number))

        key_i = key + (number - 1) if use_key else 0
        positions = data_positions((cell[0] * x, cell[1] * x), x, use_key, key_i, marker_pixels)[:y_value]
        if not positions:
            return None
        need = total_needed - len(collected)
        bits = extract_bitstream(image, positions, min(need, len(positions) * 3))
        if not bits:
            return None
        collected.extend(bits)
        number += 1

    marker_bits = 3 * (BITS_PER_CHANNEL * FIRST_CHUNK_MARKER_PIXELS + (len(used_cells) - 1) * marker_pixels * bpc)
    report = {
        "chunk_size": x, "y": y_value, "use_key": use_key,
        "grid_rows": rows, "grid_cols": cols,
        "chunks_total": grid_chunks, "chunks_used": len(used_cells), "used_cells": used_cells,
        "marker_pixels": marker_pixels, "marker_bpc": bpc, "marker_pixels_first": FIRST_CHUNK_MARKER_PIXELS,
        "payload_bits": total_needed, "header_bits": HEADER_TOTAL_BITS, "marker_bits": marker_bits,
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


def _try_decode_with_x(image: np.ndarray, analysis_matrix: np.ndarray, x: int, use_key: bool, key: int):
    order, noise_grid = chunk_rating(analysis_matrix, x, descending=True)
    grid_chunks = int(order.shape[0])
    if grid_chunks == 0:
        return None
    rows, cols = noise_grid.shape
    rank = np.empty(rows * cols, dtype=np.int64)
    rank[order[:, 0] * cols + order[:, 1]] = np.arange(grid_chunks)

    key_1 = key if use_key else 0
    grids = PixelGrids(image, x, rows, cols, BITS_PER_CHANNEL)
    rr, cc = grids.find(marker_offsets(x, use_key, key_1, FIRST_CHUNK_MARKER_PIXELS),
                        marker_value(1, FIRST_CHUNK_MARKER_PIXELS, BITS_PER_CHANNEL))
    if rr.size == 0:
        return None
    flat = rr * cols + cc
    for idx in flat[np.argsort(rank[flat], kind="stable")][:MAX_FIRST_CHUNK_CANDIDATES]:
        result = _decode_from_first_chunk(
            image, grids, x, rows, cols, rank, (int(idx) // cols, int(idx) % cols), grid_chunks, use_key, key)
        if result is not None:
            return result
    return None


def decode(image: np.ndarray, use_key: bool = False, key: int = 0, threshold: int = 8,
           x_candidates: Optional[Iterable[int]] = None) -> bytes:
    return decode_with_report(image, use_key, key, threshold, x_candidates)[0]


def decode_with_report(image: np.ndarray, use_key: bool = False, key: int = 0, threshold: int = 8,
                       x_candidates: Optional[Iterable[int]] = None, quiet: bool = False):
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
                log.info("Декодовано: x=%d, маркер %d піксел. × %d біт, %d чанків, %d байт (%s).", rep["chunk_size"],
                         rep["marker_pixels"], rep["marker_bpc"], rep["chunks_used"], rep["original_bytes"],
                         rep["compression"])
            return result
    raise ValueError(
        "Не вдалося декодувати: жоден розмір чанка x зі списку кандидатів не підійшов "
        "(перевірте key/use_key, версію кодера або розширте x_candidates)."
    )
