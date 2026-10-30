from __future__ import annotations

SUITE_NAMES = ["race", "had", "gtzan", "tau", "slue", "vesus", "desed", "vox", "vox_age"]


def load_adapter(suite: str):
    import importlib
    return importlib.import_module(f"eval.audiomarathon.adapters.{suite}")
