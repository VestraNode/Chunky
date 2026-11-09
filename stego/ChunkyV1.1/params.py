# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

from __future__ import annotations

from typing import List, Tuple

VERSION_ID = "chunky11"
VERSION_NAME = "ChunkyV1.1"

BITS_PER_CHANNEL = 1
FIRST_CHUNK_MARKER_PIXELS = 3
MAX_MARKER_PIXELS = 6
MAX_MARKER_BPC = 3
SMALL_PAYLOAD_CHUNKS = 200


def marker_value(number: int, pixels: int, bpc: int = BITS_PER_CHANNEL) -> int:
    return number % (1 << (3 * bpc * pixels))


def marker_schedule(m_start: int) -> List[Tuple[int, int]]:
    ladder = [(1, m) for m in range(max(1, m_start), MAX_MARKER_PIXELS + 1)]
    ladder += [(2, m) for m in range(3, MAX_MARKER_PIXELS + 1)]
    ladder += [(3, m) for m in range(3, MAX_MARKER_PIXELS + 1)]
    return ladder
