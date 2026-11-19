from __future__ import annotations

import re
from typing import List


def split_into_units(text: str) -> List[str]:
    """Split on periods except when they are between digits, e.g. in 21.5."""
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<!\d)\.|\.(?!\d)", text)
    units = []
    for part in parts:
        part = part.strip()
        if part:
            units.append(part + ".")
    return units
