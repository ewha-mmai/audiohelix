from __future__ import annotations

SUITE_NAMES = ["mmau"]


def load_adapter(suite: str):
    import importlib
    return importlib.import_module(f"eval.mmau_pro.adapters.{suite}")
