#!/usr/bin/env python3
"""
gb_gguf_tensor_map.py , ground-truth (name, n_bytes) tensor list for a .gguf file,
and per-expert runtime-address range generation for the hot-expert VRAM cache.

Phase 1 verification tool for the hot-expert VRAM caching plan (see
workflow/known-issues.md, "Oversized single-buffer models on Blackwell desktop
PCIe"). Used to cross-check the ordered H2D copy sequence GreenBoost records
during model load (TENSOR_LOAD_COPY events in /run/greenboost/nvtx_events.log)
against the GGUF file's actual tensor table.

Deliberately reuses the upstream llama.cpp GGUFReader (vendored at
greenboost-sources/llama.cpp/gguf-py) instead of re-deriving GGUF binary
parsing by hand , this is a one-off offline verification script, not a
runtime dependency of the shim, so correctness from reusing a known-good
parser outweighs the minor coupling to the vendored tree's layout.

Usage:
    python3 gb_gguf_tensor_map.py /path/to/model.gguf
    python3 gb_gguf_tensor_map.py /path/to/model.gguf --moe-only

    # Generate per-expert sub-ranges from runtime copy log (preferred , uses
    # actual buffer layout, not GGUF alignment estimates):
    python3 gb_gguf_tensor_map.py \\
        --from-log /run/greenboost/nvtx_events.log \\
        --ranges-file /etc/greenboost/expert_ranges.txt \\
        --n-experts 128 \\
        --emit-ranges /etc/greenboost/expert_ranges_per_expert.txt

    # Generate whole-tensor approximate ranges from GGUF (legacy, misses alignment):
    python3 gb_gguf_tensor_map.py /path/to/model.gguf \\
        --emit-ranges /etc/greenboost/expert_ranges.txt
"""
from __future__ import annotations

import re
import sys
from bisect import bisect_left
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


# ---------------------------------------------------------------------------
# Whole-tensor ranges (legacy -- cumulative sum misses alignment padding)
# ---------------------------------------------------------------------------

def expert_ranges(gguf_path: str) -> list[tuple[int, int]]:
    """Return [(offset, n_bytes), ...] for every '_exps' (MoE expert) tensor.

    NOTE: offsets are cumulative n_bytes sums and do NOT account for ggml's
    backend allocator alignment padding, so they will not match the actual
    runtime buffer offsets recorded in TENSOR_LOAD_COPY events. Use
    expert_ranges_per_expert_from_log() for ranges the shim can actually
    look up in its hash table.
    """
    offset = 0
    out: list[tuple[int, int]] = []
    for name, n_bytes in tensor_map(gguf_path):
        if "_exps" in name:
            out.append((offset, n_bytes))
        offset += n_bytes
    return out


def write_expert_ranges(gguf_path: str, out_path: str) -> int:
    """Write expert_ranges() to out_path. Returns number of ranges written."""
    ranges = expert_ranges(gguf_path)
    with open(out_path, "w") as f:
        for offset, size in ranges:
            f.write(f"{offset} {size}\n")
    return len(ranges)


# ---------------------------------------------------------------------------
# Log-driven per-expert ranges (correct , reads actual runtime buffer offsets)
# ---------------------------------------------------------------------------

def _parse_latest_load_offsets(log_path: str) -> list[int]:
    """Return all TENSOR_LOAD_COPY buffer offsets from the most recent model load.

    llama.cpp performs sequential H2D copies of tensor data into the zero-copy
    buffer.  The shim records each as a TENSOR_LOAD_COPY event with a seq
    counter that resets to 0 on every new Ollama process.  This returns the
    sorted list of unique offset values from the load starting at the last
    seq=0 occurrence (= the latest load).
    """
    offsets_this_load: list[int] = []
    in_current_load = False
    with open(log_path) as f:
        for line in f:
            if "TENSOR_LOAD_COPY" not in line:
                continue
            m_seq = re.search(r"\bseq=(\d+)", line)
            m_off = re.search(r"\boffset=(\d+)", line)
            if not (m_seq and m_off):
                continue
            seq = int(m_seq.group(1))
            off = int(m_off.group(1))
            if seq == 0:
                offsets_this_load = []
                in_current_load = True
            if in_current_load:
                offsets_this_load.append(off)
    offsets_this_load.sort()
    return offsets_this_load


def expert_ranges_per_expert_from_log(
    log_path: str,
    approx_ranges_file: str,
    n_experts: int,
) -> list[tuple[int, int]]:
    """Derive per-expert sub-ranges from the TENSOR_LOAD_COPY runtime log.

    For each whole-tensor entry in approx_ranges_file (offset, tensor_size),
    snap the offset to the nearest actual H2D copy destination recorded in the
    log (= the true runtime buffer start of that tensor), then split into
    n_experts equal-sized sub-ranges whose start addresses match the
    'src0->data + i02*nb02' pointers ggml/llama.cpp passes to cuLaunchKernel
    for each expert in a mul_mat_id dispatch (ggml-cuda.cu:2748).

    Returns [(sub_range_start_offset, per_expert_bytes), ...] for all experts
    across all _exps tensors.  The per-expert size is tensor_size / n_experts.

    Validation: the snapped tensor starts must appear as exact TENSOR_LOAD_COPY
    offsets in the log.  Mismatches (> 4 MB gap) are logged as warnings.
    """
    copy_offsets = _parse_latest_load_offsets(log_path)
    if not copy_offsets:
        raise RuntimeError(
            f"No TENSOR_LOAD_COPY events found in {log_path}. "
            "Run the model once so the shim records the copy sequence."
        )

    # Read the approximate ranges file
    approx_ranges: list[tuple[int, int]] = []
    with open(approx_ranges_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                approx_ranges.append((int(parts[0]), int(parts[1])))
    if not approx_ranges:
        raise RuntimeError(f"No ranges found in {approx_ranges_file}")

    MAX_SNAP_GAP = 4 * 1024 * 1024  # 4 MB: warn if snapped offset differs this much

    per_expert_ranges: list[tuple[int, int]] = []
    for approx_off, tensor_size in approx_ranges:
        per_expert_bytes = tensor_size // n_experts
        if per_expert_bytes * n_experts != tensor_size:
            print(
                f"# WARNING: tensor_size {tensor_size} not divisible by n_experts "
                f"{n_experts} , skipping this range",
                file=sys.stderr,
            )
            continue

        # Snap to the first copy offset >= approx_off (ceil snap, not nearest).
        # Rationale: ggml's backend allocator adds alignment padding AFTER the
        # preceding tensor, so the true tensor start is always >= the cumulative
        # size sum.  A ceil snap reliably lands on the first copy of this tensor.
        idx = bisect_left(copy_offsets, approx_off)
        if idx >= len(copy_offsets):
            print(
                f"# WARNING: no copy offset >= {approx_off} found in log "
                f"(approx range skipped)",
                file=sys.stderr,
            )
            continue
        snapped_off = copy_offsets[idx]
        gap = snapped_off - approx_off
        if gap > MAX_SNAP_GAP:
            print(
                f"# WARNING: snap gap {gap} bytes (> {MAX_SNAP_GAP}) for approx "
                f"offset {approx_off} → snapped to {snapped_off}. "
                "Ranges file may not match the loaded model.",
                file=sys.stderr,
            )

        # Emit n_experts sub-ranges: (snapped_off + i*per_expert_bytes, per_expert_bytes)
        for i in range(n_experts):
            per_expert_ranges.append((snapped_off + i * per_expert_bytes, per_expert_bytes))

    return per_expert_ranges


def write_expert_ranges_per_expert(
    log_path: str,
    approx_ranges_file: str,
    n_experts: int,
    out_path: str,
) -> int:
    """Write per-expert ranges to out_path. Returns number of lines written."""
    ranges = expert_ranges_per_expert_from_log(log_path, approx_ranges_file, n_experts)
    with open(out_path, "w") as f:
        for offset, size in ranges:
            f.write(f"{offset} {size}\n")
    return len(ranges)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    args = sys.argv[1:]

    # ------------------------------------------------------------------
    # Log-driven mode: derive per-expert ranges from runtime copy log
    # ------------------------------------------------------------------
    if "--from-log" in args:
        log_path    = args[args.index("--from-log") + 1]
        ranges_file = args[args.index("--ranges-file") + 1] if "--ranges-file" in args else None
        n_experts   = int(args[args.index("--n-experts") + 1]) if "--n-experts" in args else 128
        out_path    = args[args.index("--emit-ranges") + 1] if "--emit-ranges" in args else None
        if not ranges_file:
            print("--from-log requires --ranges-file <whole-tensor-ranges-file>",
                  file=sys.stderr)
            return 2
        if out_path:
            n = write_expert_ranges_per_expert(log_path, ranges_file, n_experts, out_path)
            print(f"# wrote {n} per-expert ranges to {out_path} "
                  f"(n_experts={n_experts}, tensors={n // n_experts})")
        else:
            ranges = expert_ranges_per_expert_from_log(log_path, ranges_file, n_experts)
            for off, sz in ranges:
                print(f"{off} {sz}")
        return 0

    # ------------------------------------------------------------------
    # GGUF file mode
    # ------------------------------------------------------------------
    path = args[0]
    moe_only = "--moe-only" in args

    if "--emit-ranges" in args:
        idx = args.index("--emit-ranges")
        out_path = args[idx + 1]
        n = write_expert_ranges(path, out_path)
        print(f"# wrote {n} expert ranges to {out_path} "
              f"(NOTE: offsets are approximate cumulative sums , "
              f"use --from-log for runtime-accurate per-expert ranges)")
        return 0

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
