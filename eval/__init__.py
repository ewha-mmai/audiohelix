from __future__ import annotations

import importlib
from types import ModuleType

BENCHES = {
    "traceav": "eval.traceav",
    "audiomarathon": "eval.audiomarathon",
    "mmau_pro": "eval.mmau_pro",
}

# the modules every benchmark must provide (see the package contract)
CONTRACT_MODULES = ("config", "data", "prompts", "evaluate", "present")


def get_bench(name: str) -> ModuleType:
    """The benchmark package, with `.config`, `.data`, `.prompts`, `.evaluate`
    and `.present` bound, so `bench.config.paths(...)` etc. resolve. The bench
    package's `__init__` re-exports them from its `setup/` and `qa/` groups,
    so this is also where a bench missing part of the contract is caught."""
    if name not in BENCHES:
        raise SystemExit(f"unknown benchmark {name!r} (choose from {', '.join(BENCHES)})")
    pkg = importlib.import_module(BENCHES[name])
    missing = [sub for sub in CONTRACT_MODULES if not hasattr(pkg, sub)]
    if missing:
        raise SystemExit(
            f"benchmark {name!r} is missing contract module(s): {', '.join(missing)}"
        )
    return pkg
