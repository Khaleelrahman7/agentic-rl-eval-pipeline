"""Loading of the external engine module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_engine(specification: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("agentic_eval_external_engine", Path(specification))
    if spec is None or spec.loader is None:
        raise ValueError("cannot load engine module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "generate_turn", None)) or not callable(getattr(module, "run_tool", None)):
        raise ValueError("engine module lacks required callables")
    return module
