#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""greenboost_bootstrap.py — the ONE thing a consumer venv needs to reach
GreenBoost's Python layer, without hardcoding a `sys.path.insert` to a dev
checkout (the gap behind 30+ ai-forge sites doing exactly that, per the
gb-synapse/gb-quant/gb-dataflux "next level" audit — only one of them even
probed the installed path).

Usage — copy this single file into any venv/environment a `.pth` drop-in
can't reach (a fresh `uv venv`, a container, ai-forge's 40+ ENV_PYTHONS
roots), or `pip install -e` it, then:

    import greenboost_bootstrap   # adds the GreenBoost root to sys.path
    import gb_api                 # or gb_quant, gb_cluster, gb_synapse, ...

Deliberately self-contained — stdlib only, no `import gb_paths`. This
file's whole job is running BEFORE gb_paths.py's own directory is on
sys.path, so it cannot depend on anything inside that directory; the tiny
resolution logic below is intentionally a duplicate of gb_paths.gb_py_root
for that reason, not an oversight.
"""
import os
import sys
from pathlib import Path

_INSTALLED_ROOT = Path("/usr/local/lib/greenboost")
_DEV_ROOT = Path.home() / "Dev" / "greenboost_all" / "greenboost"


def _gb_py_root() -> Path:
    env = os.environ.get("GB_PY_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    if (_INSTALLED_ROOT / "gb_synapse.py").is_file():
        return _INSTALLED_ROOT
    return _DEV_ROOT


_root = str(_gb_py_root())
if _root not in sys.path:
    sys.path.insert(0, _root)
