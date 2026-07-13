"""gb_paths.py — single resolver for the GreenBoost python root.

Every place greenboost-cli needs the sibling greenboost checkout
(gb_synapse, gb_quant, gb_monitor, gb_pilot…) goes through here instead
of hardcoding a path.

Resolution order:
  1. env GB_PY_ROOT — explicit override (the installed `gb` /
     `greenboost-cli` wrappers set it)
  2. /usr/local/lib/greenboost — the Full Install destination, used only
     when it actually contains gb_synapse.py (a half-installed root must
     never shadow a working dev checkout)
  3. ~/Dev/greenboost_all/greenboost — dev checkout fallback
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_INSTALLED_ROOT = Path("/usr/local/lib/greenboost")
_DEV_ROOT = Path.home() / "Dev/greenboost_all/greenboost"


def gb_py_root() -> Path:
    """Resolve the GreenBoost python root (see module docstring for order)."""
    env = os.environ.get("GB_PY_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    if (_INSTALLED_ROOT / "gb_synapse.py").is_file():
        return _INSTALLED_ROOT
    return _DEV_ROOT


def gb_module(name: str):
    """Import module *name* from the GreenBoost python root."""
    root = str(gb_py_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def gb_root_hint() -> str:
    """User-facing hint for when the GreenBoost python root is missing."""
    return (
        "set GB_PY_ROOT, or run greenboost Full Install (installs to "
        "/usr/local/lib/greenboost), or clone to ~/Dev/greenboost_all/greenboost"
    )
