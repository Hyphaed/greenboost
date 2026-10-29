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

import os
import re
import sys
from pathlib import Path

_VENDORED_GGUF_PY = Path(__file__).parent.parent / "greenboost-sources" / "llama.cpp" / "gguf-py"

# llama.cpp's own split-shard naming (gguf-py/gguf/gguf_writer.py,
# SHARD_NAME_FORMAT). Canonical home for this regex + shard_paths(): this
# module has no dependency on gb_synapse.py, while gb_synapse.py already
# imports FROM this module (_load_gguf_reader) — so gb_synapse.py imports
# shard_paths() from HERE rather than each keeping its own copy. Real
# incident, 2026-08-28: this module's tensor_map() (used by
# _fit_cpu_moe_layers() to size MoE --n-cpu-moe offload) summed only the one
# file it was handed, exactly the same blind spot gb_synapse.gguf_summary()
# had before its split-aware fix — for Qwen3.8-Flash-Next-UD-Q2_K_XL that
# meant `expert_bytes: 0` from this function specifically, `_fit_cpu_moe_
# layers()` returning 0 (nothing to offload), no `--n-cpu-moe` flag emitted
# at all, and llama.cpp requesting one ~46 GiB CUDA buffer for the experts
# — a real cudaMalloc OOM the shim's front-load split was never given a
# chance to handle, because llama.cpp never asked it to.
_GGUF_SPLIT_RE = re.compile(r'^(.+)-(\d{5})-of-(\d{5})\.gguf$')


def shard_paths(path: str) -> list[str]:
    """Every sibling shard of a split GGUF, in shard order, derived from the
    `-%05d-of-%05d.gguf` filename convention -- `[path]` unchanged for a
    non-split file. Never a directory glob: a glob would pick up an
    unrelated second quant sitting in the same cache folder."""
    m = _GGUF_SPLIT_RE.match(os.path.basename(path))
    if not m:
        return [path]
    stem, _, total = m.groups()
    d = os.path.dirname(path)
    return [os.path.join(d, f"{stem}-{i:05d}-of-{total}.gguf")
            for i in range(1, int(total) + 1)]


def _load_gguf_reader():
    if str(_VENDORED_GGUF_PY) not in sys.path:
        sys.path.insert(0, str(_VENDORED_GGUF_PY))
    from gguf.gguf_reader import GGUFReader  # noqa: E402
    return GGUFReader


def tensor_map(gguf_path: str) -> list[tuple[str, int]]:
    """Return [(tensor_name, n_bytes), ...] in on-disk declaration order,
    across every shard of a split GGUF (see shard_paths()'s incident note
    above for why this matters). A missing sibling shard raises rather than
    silently returning a partial tensor list — the same fail-loud contract
    gb_synapse.gguf_summary() uses for the identical failure mode."""
    paths = shard_paths(gguf_path)
    if len(paths) > 1:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"tensor_map: split GGUF {gguf_path!r} is missing "
                f"{len(missing)}/{len(paths)} sibling shard(s): "
                f"{', '.join(missing)}")
    GGUFReader = _load_gguf_reader()
    out: list[tuple[str, int]] = []
    for p in paths:
        reader = GGUFReader(p, mode="r")
        out.extend((t.name, int(t.n_bytes)) for t in reader.tensors)
    return out


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
