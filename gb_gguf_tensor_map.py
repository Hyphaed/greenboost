#!/usr/bin/env python3
"""
gb_gguf_tensor_map.py , ground-truth (name, n_bytes) tensor list for a .gguf file.

Deliberately reuses the upstream llama.cpp GGUFReader (vendored at
greenboost-sources/llama.cpp/gguf-py) instead of re-deriving GGUF binary
parsing by hand , this is a one-off offline inspection helper, not a
runtime dependency of the shim, so correctness from reusing a known-good
parser outweighs the minor coupling to the vendored tree's layout.

Also exposes _load_gguf_reader() / tensor_map(), imported by gb_synapse.py
for GGUF metadata summaries.

Usage:
    python3 gb_gguf_tensor_map.py /path/to/model.gguf
    python3 gb_gguf_tensor_map.py /path/to/model.gguf --moe-only
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDORED_GGUF_PY = Path(__file__).parent.parent / "greenboost-sources" / "llama.cpp" / "gguf-py"


def _load_gguf_reader():
    if str(_VENDORED_GGUF_PY) not in sys.path:
        sys.path.insert(0, str(_VENDORED_GGUF_PY))
    from gguf.gguf_reader import GGUFReader  # noqa: E402
    return GGUFReader


def tensor_map(gguf_path: str) -> list[tuple[str, int]]:
    """Return [(tensor_name, n_bytes), ...] in on-disk declaration order."""
    GGUFReader = _load_gguf_reader()
    reader = GGUFReader(gguf_path, mode="r")
    return [(t.name, int(t.n_bytes)) for t in reader.tensors]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    args = sys.argv[1:]

    path = args[0]
    moe_only = "--moe-only" in args

    entries = tensor_map(path)
    total = sum(n for _, n in entries)
    print(f"# {path}")
    print(f"# {len(entries)} tensors, {total / (1024**3):.2f} GiB total")
    print(f"#{'name':<48} {'bytes':>14} {'MiB':>10}")
    for name, n_bytes in entries:
        if moe_only and "_exps" not in name:
            continue
        print(f"{name:<48} {n_bytes:>14} {n_bytes / (1024**2):>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
