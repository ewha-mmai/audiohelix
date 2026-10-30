from __future__ import annotations

from typing import List


def split_into_units(text: str) -> List[str]:
    """"A dog barks. Music plays softly." -> ["A dog barks.", "Music plays softly."]"""
    if not text or not text.strip():
        return []
    parts = text.split(".")
    units = []
    for part in parts:
        part = part.strip()
        if part:
            units.append(part + ".")
    return units
