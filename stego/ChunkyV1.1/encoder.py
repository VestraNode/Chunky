# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

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
    bytes_to_bits,
    data_pixels_per_chunk,
    int_to_bits,
)
from ..compression import METHOD_NAMES, compress_best
from ..lsb_match import apply_low_bits, apply_lsb_matching, toggle_costs
from .decoder import BPC_FIELD_BITS, METHOD_FIELD_BITS, candidates_for, decode_with_report, ranked_value_map
from .layout import PixelGrids, data_positions, marker_offsets, ordered_positions
from .params import (
    BITS_PER_CHANNEL,
    FIRST_CHUNK_MARKER_PIXELS,
    SMALL_PAYLOAD_CHUNKS,
    marker_schedule,
    marker_value,
)

log = logging.getLogger("stego.chunky.encoder")

MAX_CYCLES = 30


class _MarkerConflict(ValueError):
    pass


def capacity_bytes(num_chunks: int, x: int, y: int, use_key: bool) -> int:
    if num_chunks <= 0:
        return 0
    first = data_pixels_per_chunk(x, use_key, FIRST_CHUNK_MARKER_PIXELS)
    other = min(y, data_pixels_per_chunk(x, use_key, 3))
    if first <= 0 or other < 0:
        return 0
    return max(0, ((first + (num_chunks - 1) * other) * 3 - HEADER_TOTAL_BITS) // 8)


def _chunks_needed(total_bits: int, x: int, y: int, use_key: bool, m: int) -> int:
    first = data_pixels_per_chunk(x, use_key, FIRST_CHUNK_MARKER_PIXELS)
    other = min(y, data_pixels_per_chunk(x, use_key, m))
    remaining = total_bits - first * 3
    if remaining <= 0:
        return 1
    if other <= 0:
        return 0
    return 1 + -(-remaining // (other * 3))


def _assign(order: np.ndarray, x: int, y: int, use_key: bool, key: int, m: int, bpc: int,
            full_bits: List[int]):
    total, cursor, number = len(full_bits), 0, 1
    infos: List[dict] = []
    singles: List[Tuple[np.ndarray, ...]] = []
    multis: List[Tuple[np.ndarray, ...]] = []

    for (r, c) in order:
        if cursor >= total:
            break
        marker_pixels = FIRST_CHUNK_MARKER_PIXELS if number == 1 else m
        chunk_bpc = BITS_PER_CHANNEL if number == 1 else bpc
        top_left = (int(r) * x, int(c) * x)
        key_i = key + (number - 1) if use_key else 0
        ordered = ordered_positions(top_left, x, use_key, key_i)
        if len(ordered) <= marker_pixels:
            continue

        value = marker_value(number, marker_pixels, chunk_bpc)
        mask = (1 << chunk_bpc) - 1
        mr, mc, mch, mvals = [], [], [], []
        for j, (py, px) in enumerate(ordered[:marker_pixels]):
            part = (value >> (3 * chunk_bpc * (marker_pixels - 1 - j))) & ((1 << (3 * chunk_bpc)) - 1)
            mr += [py] * 3; mc += [px] * 3; mch += [0, 1, 2]
            mvals += [(part >> (2 * chunk_bpc)) & mask, (part >> chunk_bpc) & mask, part & mask]
        group = (np.array(mr), np.array(mc), np.array(mch), np.array(mvals))
        (singles if chunk_bpc == 1 else multis).append(group)

        data_pos = data_positions(top_left, x, use_key, key_i, marker_pixels)
        if number != 1:
            data_pos = data_pos[:y]
        chunk_bits = full_bits[cursor: cursor + 3 * len(data_pos)]
        length = len(chunk_bits)
        if length:
            pos = np.array(data_pos[: -(-length // 3)])
            singles.append((np.repeat(pos[:, 0], 3)[:length], np.repeat(pos[:, 1], 3)[:length],
                            np.tile([0, 1, 2], len(pos))[:length], np.array(chunk_bits)))
        cursor += length

        infos.append({
            "number": number, "cell": (int(r), int(c)),
            "offsets": [(py - top_left[0], px - top_left[1]) for (py, px) in ordered[:marker_pixels]],
        })
        number += 1

    if cursor < total:
        raise ValueError(
            f"Недостатньо місткості для приховування даних: записано {cursor} з {total} біт. "
            f"Збільште x, y, увімкніть архівацію або зменште обсяг даних.")

    def stack(parts):
        return [np.concatenate([p[i] for p in parts]).astype(np.int64) for i in range(4)] if parts else None

    return infos, stack(singles), stack(multis)


def _used_conflict(img: np.ndarray, x: int, infos: List[dict], rank: np.ndarray, cols: int, m: int, bpc: int):
    rest = infos[1:]
    if not rest:
        return None
    count = len(rest)
    mask = (1 << bpc) - 1
    ur = np.array([i["cell"][0] for i in rest])
    uc = np.array([i["cell"][1] for i in rest])
    urank = rank[ur * cols + uc]
    index = np.arange(count)
    ar = np.arange(x)
    blocks = img[ur[:, None, None] * x + ar[None, :, None], uc[:, None, None] * x + ar[None, None, :]].astype(np.int64) & mask
    codes = ((blocks[..., 0] << (2 * bpc)) | (blocks[..., 1] << bpc) | blocks[..., 2]).reshape(count, x * x)
    for i, info in enumerate(rest):
        value = marker_value(info["number"], m, bpc)
        values = np.zeros(count, dtype=np.int64)
        for (dy, dx) in info["offsets"]:
            values = (values << (3 * bpc)) | codes[:, dy * x + dx]
        if np.any((values == value) & (index > i) & (urank < urank[i])):
            return f"маркер чанка №{info['number']} збігається з чанком, що містить інформацію"
    return None


def _resolve_interference(img: np.ndarray, origin: np.ndarray, x: int, grid_shape, infos: List[dict],
                          use_key: bool, key: int, m: int, bpc: int, threshold: int):
    rows, cols = grid_shape
    used_cells = {info["cell"] for info in infos}
    first_idx = infos[0]["cell"][0] * cols + infos[0]["cell"][1]
    fixes = 0
    touched: set = set()
    used_values = None if use_key else {marker_value(info["number"], m, bpc) for info in infos[1:]}

    for cycle in range(1, MAX_CYCLES + 1):
        analysis = steganalysis(img, threshold)
        order, _ = chunk_rating(analysis, x, descending=True)
        rank = np.empty(rows * cols, dtype=np.int64)
        rank[order[:, 0] * cols + order[:, 1]] = np.arange(order.shape[0])

        if use_key:
            conflict = _used_conflict(img, x, infos, rank, cols, m, bpc)
            if conflict:
                raise _MarkerConflict(conflict)

        value_map = None
        if not use_key:
            value_map = ranked_value_map(img, x, rows, cols, marker_offsets(x, False, 0, m), rank, bpc)
        grids = PixelGrids(img, x, rows, cols, bpc)

        taken = np.zeros(rows * cols, dtype=bool)
        taken[first_idx] = True
        interferers: Dict[Tuple[Tuple[int, int], int], list] = {}

        for info in infos[1:]:
            number, true_cell = info["number"], info["cell"]
            true_idx = true_cell[0] * cols + true_cell[1]
            cand = candidates_for(grids, x, cols, rank, number, m, use_key, key, value_map)
            for idx in cand:
                idx = int(idx)
                if idx == true_idx:
                    break
                if taken[idx]:
                    continue
                cell = (idx // cols, idx % cols)
                if cell in used_cells:
                    raise _MarkerConflict(f"маркер чанка №{number} збігається з чанком, що містить інформацію")
                interferers[(cell, number)] = info["offsets"]
            else:
                raise _MarkerConflict(f"маркер чанка №{number} не знайдено після запису")
            taken[true_idx] = True

        if not interferers:
            return fixes, cycle

        cand_r, cand_c, cand_k, owner, allowed = [], [], [], [], []
        keys = list(interferers)
        for oi, (cell, number) in enumerate(keys):
            value = marker_value(number, m, bpc)
            for j, (dy, dx) in enumerate(interferers[(cell, number)]):
                for chn in range(3):
                    r, c = cell[0] * x + dy, cell[1] * x + dx
                    ok = (r, c, chn) not in touched
                    if ok and used_values is not None:
                        shift = 3 * bpc * (m - 1 - j) + (2 - chn) * bpc
                        ok = (value ^ (1 << shift)) not in used_values
                    cand_r.append(r); cand_c.append(c); cand_k.append(chn); owner.append(oi); allowed.append(ok)
        cand_r, cand_c, cand_k, owner = map(np.array, (cand_r, cand_c, cand_k, owner))
        costs = toggle_costs(img, cand_r, cand_c, cand_k, threshold, origin)
        costs[~np.array(allowed)] = np.inf
        chosen = set()
        for oi in range(len(keys)):
            idxs = np.nonzero(owner == oi)[0]
            best = idxs[int(np.argmin(costs[idxs]))]
            if not np.isfinite(costs[best]):
                raise _MarkerConflict(f"немає вільного каналу, щоб усунути збіг маркера чанка №{keys[oi][1]}")
            chosen.add((int(cand_r[best]), int(cand_c[best]), int(cand_k[best])))
        touched |= chosen
        rr, cc, kk = (np.array(a) for a in zip(*sorted(chosen)))
        apply_lsb_matching(img, rr, cc, kk, (img[rr, cc, kk] & 1) ^ 1, threshold, origin)
        fixes += len(chosen)
        log.info("Перевірка, цикл %d: заважають %d чанків — змінено по 1 каналу в їхніх маркерах.",
                 cycle, len(keys))

    raise _MarkerConflict(f"збіги маркерів не усунено за {MAX_CYCLES} циклів")


def encode(image: np.ndarray, x: int, y: int, data: bytes, use_key: bool = False, key: int = 0,
           threshold: int = 8, compress: bool = True, optimize_markers: bool = False) -> np.ndarray:
    return encode_with_report(image, x, y, data, use_key, key, threshold, compress, optimize_markers)[0]


def encode_with_report(image: np.ndarray, x: int, y: int, data: bytes, use_key: bool = False,
                       key: int = 0, threshold: int = 8, compress: bool = True,
                       optimize_markers: bool = False):
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
            log.info("Архівація: %s, %d -> %d байт (%.1f%% від початкового)", METHOD_NAMES[method],
                     original_bytes, len(stored), 100 * len(stored) / max(1, original_bytes))
        else:
            log.warning("Архівація не дала виграшу (%s) — зберігаю без стиснення.",
                        ", ".join(f"{k}: {v}" for k, v in candidates.items()) or "дані порожні")
    else:
        method, stored, candidates = 0, data, {}

    payload_bits = bytes_to_bits(stored)
    header_base = (
        int_to_bits(MAGIC_BYTE, MAGIC_LEN_BITS) + int_to_bits(len(payload_bits), LENGTH_FIELD_BITS)
        + int_to_bits(y, Y_FIELD_BITS)
    )
    total_bits = HEADER_TOTAL_BITS + len(payload_bits)

    chunks_at_3 = _chunks_needed(total_bits, x, y, use_key, 3)
    if chunks_at_3 == 0 or chunks_at_3 > grid_chunks:
        raise ValueError(
            f"Недостатньо місткості для приховування даних ({total_bits} біт разом із заголовком). "
            f"Збільште x, y, увімкніть архівацію або зменште обсяг даних.")
    m_plan = 2 if chunks_at_3 <= SMALL_PAYLOAD_CHUNKS else 3
    log.info("Маркер: план M=%d пікселів (чанків потрібно ≈%d із %d, чанк №1 — %d пікселі).",
             m_plan, chunks_at_3, grid_chunks, FIRST_CHUNK_MARKER_PIXELS)

    schedule = marker_schedule(1 if optimize_markers else m_plan)
    if use_key and not optimize_markers:
        need_bits = math.ceil(math.log2(max(2.0, chunks_at_3 * chunks_at_3 / 2 / 15)))
        usable = [(b, m) for (b, m) in schedule if 3 * b * m >= need_bits]
        if usable and len(usable) < len(schedule):
            log.info("Багато чанків (%d) з ключем: починаю одразу з маркера %d піксел. × %d біт.",
                     chunks_at_3, usable[0][1], usable[0][0])
            schedule = usable

    result = None
    trials: List[Tuple[str, str]] = []
    for attempt, (bpc, m) in enumerate(schedule):
        label = f"M={m}×{bpc} біт"
        n_est = _chunks_needed(total_bits, x, y, use_key, m)
        if n_est == 0 or n_est > grid_chunks:
            trials.append((label, "не вистачає місткості"))
            continue
        if bpc > 1 and attempt > 0 and schedule[attempt - 1][0] < bpc:
            log.warning("Маркер по 1 біту не виходить — повертаюсь до M=%d, але по %d біти на канал.", m, bpc)
        header = header_base + int_to_bits(m, MARKER_COUNT_FIELD_BITS) \
            + int_to_bits(bpc - 1, BPC_FIELD_BITS) + int_to_bits(method, METHOD_FIELD_BITS)
        try:
            infos, singles, multis = _assign(order, x, y, use_key, key, m, bpc, header + payload_bits)
            img = image.copy()
            lsb_changes = apply_lsb_matching(img, singles[0], singles[1], singles[2], singles[3], threshold, image)
            if multis is not None:
                apply_low_bits(img, multis[0], multis[1], multis[2], multis[3], bpc)
            fixes, cycles = _resolve_interference(img, image, x, noise_grid.shape, infos, use_key, key, m, bpc, threshold)
            try:
                check, _ = decode_with_report(img, use_key, key, threshold, x_candidates=[x], quiet=True)
            except ValueError:
                check = None
            if check != data:
                raise _MarkerConflict("результат не проходить перевірку декодуванням")
        except _MarkerConflict as exc:
            trials.append((label, str(exc)))
            log.warning("%s: %s — наступна спроба маркера.", label, exc)
            continue
        result = (m, bpc, img, infos, lsb_changes, fixes, cycles)
        break

    if result is None:
        raise ValueError("Не вдалося підібрати маркер (" + "; ".join(f"{lab}: {why}" for lab, why in trials)
                         + "). Спробуйте інший ключ або x.")
    m, bpc, img, infos, lsb_changes, fixes, cycles = result
    if optimize_markers:
        log.info("Оптимізація: мінімальний робочий маркер — %d піксел. × %d біт (за планом було %d).", m, bpc, m_plan)
    if bpc > 1:
        log.warning("Маркер записано по %d біти на канал: у пікселях маркерів значення можуть змінюватись до ±%d.",
                    bpc, (1 << bpc) - 1)
    if fixes:
        log.warning("Усунено %d збігів маркерів за %d цикл(ів) (по 1 каналу ±1 у сторонніх чанках).", fixes, cycles)
    log.info("LSB matching: %d значень каналів змінено на ±1 у бік меншого шуму.", lsb_changes)

    used = {info["cell"]: info["number"] for info in infos}
    marker_bits = 3 * (BITS_PER_CHANNEL * FIRST_CHUNK_MARKER_PIXELS + (len(infos) - 1) * m * bpc)
    report = {
        "chunk_size": x, "y": y, "use_key": use_key,
        "grid_rows": int(noise_grid.shape[0]), "grid_cols": int(noise_grid.shape[1]),
        "chunks_total": grid_chunks, "chunks_used": len(infos),
        "used_cells": [(r, c, n) for (r, c), n in used.items()],
        "marker_pixels": m, "marker_bpc": bpc, "marker_pixels_safe": m_plan,
        "marker_pixels_first": FIRST_CHUNK_MARKER_PIXELS,
        "optimized": optimize_markers,
        "compression": METHOD_NAMES[method], "compression_method": method,
        "compression_candidates": candidates,
        "original_bytes": original_bytes, "stored_bytes": len(stored),
        "payload_bits": len(payload_bits), "header_bits": HEADER_TOTAL_BITS, "marker_bits": marker_bits,
        "used_bits_total": len(payload_bits) + HEADER_TOTAL_BITS + marker_bits,
        "capacity_bytes": capacity_bytes(grid_chunks, x, y, use_key),
        "marker_fixes": fixes, "cycles": cycles, "lsb_changes": lsb_changes,
    }
    log.info("Закодовано: %d чанків із %d, маркер %d піксел. × %d біт, збережено %d байт.", len(infos), grid_chunks,
             m, bpc, len(stored))
    return img, report
