# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Optional, List, Sequence, Tuple

import numpy as np

MAGIC_BYTE = 0xC3

MARKER_BITS_PER_CHANNEL = 3

MAGIC_LEN_BITS = 8
LENGTH_FIELD_BITS = 32
Y_FIELD_BITS = 32
MARKER_COUNT_FIELD_BITS = 4
COMPRESSION_FIELD_BITS = 4
HEADER_TOTAL_BITS = (MAGIC_LEN_BITS + LENGTH_FIELD_BITS + Y_FIELD_BITS
                     + MARKER_COUNT_FIELD_BITS + COMPRESSION_FIELD_BITS)

FIRST_CHUNK_MARKER_PIXELS = 2
MAX_MARKER_PIXELS = 6
MARKER_SAFETY_BITS = 12


def bytes_to_bits(data: bytes) -> List[int]:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8)).tolist()


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    arr = np.asarray(list(bits), dtype=np.int64) & 1
    pad = (-arr.size) % 8
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.int64)])
    return np.packbits(arr.astype(np.uint8)).tobytes()


def int_to_bits(value: int, n_bits: int) -> List[int]:
    return [(value >> shift) & 1 for shift in range(n_bits - 1, -1, -1)]


def bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | (int(b) & 1)
    return value


def embed_bits_in_pixel(pixel: np.ndarray, bits: Sequence[int]) -> int:
    n = min(3, len(bits))
    for c in range(n):
        pixel[c] = (int(pixel[c]) & ~1) | (int(bits[c]) & 1)
    return n


def extract_bits_from_pixel(pixel: np.ndarray, n_bits: int) -> List[int]:
    n = min(3, n_bits)
    return [int(pixel[c]) & 1 for c in range(n)]


def embed_value_in_pixel(pixel: np.ndarray, value: int, bits_per_channel: int) -> None:
    total_bits = bits_per_channel * 3
    bits = int_to_bits(value, total_bits)
    mask = (1 << bits_per_channel) - 1
    for c in range(3):
        chunk_bits = bits[c * bits_per_channel:(c + 1) * bits_per_channel]
        chunk_val = bits_to_int(chunk_bits)
        pixel[c] = (int(pixel[c]) & ~mask) | (chunk_val & mask)


def extract_value_from_pixel(pixel: np.ndarray, bits_per_channel: int) -> int:
    mask = (1 << bits_per_channel) - 1
    bits: List[int] = []
    for c in range(3):
        chunk_val = int(pixel[c]) & mask
        bits.extend(int_to_bits(chunk_val, bits_per_channel))
    return bits_to_int(bits)


def embed_marker(image: np.ndarray, positions: Sequence[Tuple[int, int]], value: int,
                  bits_per_channel: int = MARKER_BITS_PER_CHANNEL) -> None:
    bits_per_pixel = bits_per_channel * 3
    if value >> (bits_per_pixel * len(positions)):
        raise ValueError(
            f"Номер чанка {value} не вміщується у маркер із {len(positions)} пікселів."
        )
    pixel_mask = (1 << bits_per_pixel) - 1
    for j, (py, px) in enumerate(positions):
        part = (value >> (bits_per_pixel * (len(positions) - 1 - j))) & pixel_mask
        embed_value_in_pixel(image[py, px], part, bits_per_channel)


def extract_marker(image: np.ndarray, positions: Sequence[Tuple[int, int]],
                    bits_per_channel: int = MARKER_BITS_PER_CHANNEL) -> int:
    value = 0
    for (py, px) in positions:
        value = (value << (bits_per_channel * 3)) | extract_value_from_pixel(image[py, px], bits_per_channel)
    return value


def _pixel_marker_values(image: np.ndarray, size: int, rows: int, cols: int,
                          offset: Tuple[int, int], bits_per_channel: int,
                          cache: Optional[dict] = None) -> np.ndarray:
    key = (offset, bits_per_channel)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    mask = (1 << bits_per_channel) - 1
    dr, dc = offset
    sub = image[dr: rows * size: size, dc: cols * size: size] & mask
    values = ((sub[..., 0].astype(np.int64) << (2 * bits_per_channel)) | (sub[..., 1].astype(np.int64) << bits_per_channel) | sub[..., 2])
    if cache is not None:
        cache[key] = values
    return values


def invalidate_marker_cache(cache: Optional[dict], py: int, px: int, size: int) -> None:
    if cache:
        for bpc in {k[1] for k in cache}:
            cache.pop(((py % size, px % size), bpc), None)


def find_marker_cells(image: np.ndarray, size: int, rows: int, cols: int,
                       offsets: Sequence[Tuple[int, int]], value: int,
                       bits_per_channel: int = MARKER_BITS_PER_CHANNEL, cache: Optional[dict] = None):
    bits_per_pixel = bits_per_channel * 3
    empty = np.zeros(0, dtype=np.int64)
    if value >> (bits_per_pixel * len(offsets)):
        return empty, empty
    pixel_mask = (1 << bits_per_pixel) - 1
    parts = [(value >> (bits_per_pixel * (len(offsets) - 1 - j))) & pixel_mask for j in range(len(offsets))]

    first = _pixel_marker_values(image, size, rows, cols, offsets[0], bits_per_channel, cache)
    rr, cc = np.nonzero(first == parts[0])
    ch_mask = (1 << bits_per_channel) - 1
    for j in range(1, len(offsets)):
        if rr.size == 0:
            break
        dr, dc = offsets[j]
        sub = image[rr * size + dr, cc * size + dc].astype(np.int64) & ch_mask
        pv = (sub[:, 0] << (2 * bits_per_channel)) | (sub[:, 1] << bits_per_channel) | sub[:, 2]
        keep = pv == parts[j]
        rr, cc = rr[keep], cc[keep]
    return rr, cc


def marker_value_map(image: np.ndarray, size: int, rows: int, cols: int,
                      offsets: Sequence[Tuple[int, int]],
                      bits_per_channel: int = MARKER_BITS_PER_CHANNEL) -> dict:
    values = np.zeros((rows, cols), dtype=np.int64)
    for off in offsets:
        values = (values << (bits_per_channel * 3)) | _pixel_marker_values(
            image, size, rows, cols, off, bits_per_channel)
    result: dict = {}
    for idx, v in enumerate(values.ravel().tolist()):
        result.setdefault(v, []).append(idx)
    return result


def marker_pixels_needed(grid_chunks: int, used_chunks: int,
                          bits_per_channel: int = MARKER_BITS_PER_CHANNEL) -> int:
    need_bits = math.ceil(math.log2(max(2, grid_chunks) * max(2, used_chunks))) + MARKER_SAFETY_BITS
    return max(FIRST_CHUNK_MARKER_PIXELS, math.ceil(need_bits / (bits_per_channel * 3)))


def data_pixels_per_chunk(size: int, use_key: bool, marker_pixels: int) -> int:
    pixels = size * size
    return max(0, pixels - marker_pixels) if use_key else len(range(marker_pixels, pixels, 2))


def chunks_needed(total_bits: int, size: int, y: int, use_key: bool, marker_pixels: int) -> int:
    first = data_pixels_per_chunk(size, use_key, FIRST_CHUNK_MARKER_PIXELS)
    other = min(y, data_pixels_per_chunk(size, use_key, marker_pixels))
    remaining = total_bits - first * 3
    if remaining <= 0:
        return 1
    if other <= 0:
        return 0
    return 1 + -(-remaining // (other * 3))


def plan_marker_pixels(grid_chunks: int, total_bits: int, size: int, y: int,
                        use_key: bool) -> Tuple[int, int]:
    m = FIRST_CHUNK_MARKER_PIXELS
    while m <= MAX_MARKER_PIXELS:
        n = chunks_needed(total_bits, size, y, use_key, m)
        if n == 0 or n > grid_chunks:
            return m, 0
        needed_m = marker_pixels_needed(grid_chunks, n)
        if needed_m <= m:
            return m, n
        m = needed_m
    return MAX_MARKER_PIXELS, 0


def key_shuffle_order(n: int, key: int) -> List[int]:
    return _shuffle_cached(n, key).tolist()


@lru_cache(maxsize=2048)
def _shuffle_cached(n: int, key: int) -> np.ndarray:
    rng = random.Random(key)
    order = list(range(n))
    rng.shuffle(order)
    return np.array(order, dtype=np.int32)


def chunk_pixel_list(top_left: Tuple[int, int], size: int) -> List[Tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr in range(size) for dc in range(size)]


def ordered_chunk_positions(top_left: Tuple[int, int], size: int, use_key: bool,
                             key: int) -> List[Tuple[int, int]]:
    pixels = chunk_pixel_list(top_left, size)
    if use_key:
        order = key_shuffle_order(len(pixels), key)
        return [pixels[i] for i in order]
    return pixels


def data_positions_for_chunk(top_left: Tuple[int, int], size: int, use_key: bool,
                              key: int, marker_pixels: int = FIRST_CHUNK_MARKER_PIXELS
                              ) -> List[Tuple[int, int]]:
    if use_key:
        ordered = ordered_chunk_positions(top_left, size, use_key=True, key=key)
        return ordered[marker_pixels:]
    pixels = chunk_pixel_list(top_left, size)
    return pixels[marker_pixels::2]


def embed_bitstream(image: np.ndarray, positions: Sequence[Tuple[int, int]],
                     bits: Sequence[int]) -> int:
    total = len(bits)
    n_pix = min(len(positions), -(-total // 3))
    if n_pix <= 0:
        return 0
    pos = np.asarray(positions[:n_pix], dtype=np.int64).reshape(n_pix, 2)
    n = min(total, 3 * n_pix)
    values = np.asarray(list(bits[:n]), dtype=np.int64) & 1
    rows = np.repeat(pos[:, 0], 3)[:n]
    cols = np.repeat(pos[:, 1], 3)[:n]
    chans = np.tile(np.arange(3), n_pix)[:n]
    image[rows, cols, chans] = ((image[rows, cols, chans] & 0xFE) | values).astype(image.dtype)
    return n


def extract_bitstream(image: np.ndarray, positions: Sequence[Tuple[int, int]],
                       n_bits: int) -> List[int]:
    n_pix = min(len(positions), -(-max(0, n_bits) // 3))
    if n_pix <= 0:
        return []
    pos = np.asarray(positions[:n_pix], dtype=np.int64).reshape(n_pix, 2)
    lsb = image[pos[:, 0], pos[:, 1]] & 1
    return lsb.reshape(-1)[:n_bits].astype(np.int64).tolist()
