"""
GreenBoost slash commands.

/turboquant [on|off|status]  — toggle or query TurboQuant KV compression
/gb-status                   — show GreenBoost T1/T2/T3 tier statistics
/gb-tiers                    — detailed tier breakdown
/gb-vitals                   — full debug vitals including inference server diagnostics
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from greenboost_cli.terminal.theme import emit_ok, emit_err, emit_warn, emit_info, console, VIOLET, GRAY, LIME, AMBER

_GB_FLAG_DIR = Path("/etc/greenboost")
_GB_TQ_FLAG  = _GB_FLAG_DIR / "turboquant.enabled"
_GB_CLUSTER  = _GB_FLAG_DIR / "cluster.conf"


def _gb_installed() -> bool:
    return Path("/sys/module/greenboost").exists() or _GB_FLAG_DIR.exists()


def _tq_state() -> bool:
    return _GB_TQ_FLAG.exists() or os.environ.get("GREENBOOST_TURBOQUANT") == "1"


def cmd_turboquant(args: str, _session, _settings) -> bool:
    subcmd = args.strip().lower() or "status"

    if subcmd == "status":
        if not _gb_installed():
            emit_warn("GreenBoost is not installed.")
            return True
        tq_on = _tq_state()
        feeders = 0
        try:
            feeders = sum(1 for ln in _GB_CLUSTER.read_text().splitlines() if ln.strip())
        except Exception:
            pass
        state_str = f"[{LIME}]ON[/]" if tq_on else f"[dim {GRAY}]OFF[/]"
        console.print(f"\n  [{GRAY}]TurboQuant:[/]  {state_str}")
        console.print(f"  [{GRAY}]Cluster:   [/]  [{GRAY}]{feeders} feeder(s)[/]")
        env_val = os.environ.get("GREENBOOST_TURBOQUANT", "")
        if env_val == "1":
            console.print(f"  [{GRAY}]Env var:   [/]  [{LIME}]GREENBOOST_TURBOQUANT=1[/]")
        console.print()
        return True

    if subcmd in ("on", "fp8", "3bit", "off"):
        if not _gb_installed():
            emit_warn("GreenBoost is not installed — cannot toggle TurboQuant.")
            return True
        enabled = subcmd != "off"
        bits = 3 if subcmd == "3bit" else 8   # FP8 (8-bit) is default; /turboquant 3bit for legacy
        # Prefer IOCTL path (no sudo needed, immediate effect)
        try:
            from greenboost_cli.greenboost.monitor import get_monitor
            m = get_monitor()
            ok = m.set_turboquant(enabled=enabled, bits=bits)
        except Exception:
            ok = False
        if not ok:
            # Fallback to CLI helper
            result = subprocess.run(["sudo", "greenboost", "turboquant", subcmd])
            ok = result.returncode == 0
        if ok:
            os.environ["GREENBOOST_TURBOQUANT"] = "1" if enabled else ""
            label = f"FP8 KV ({bits}-bit)" if enabled else "OFF"
            emit_ok(f"TurboQuant {label}")
        return True

    emit_err(f"Unknown /turboquant subcommand: '{subcmd}'  (use: on | fp8 | 3bit | off | status)")
    return True


def cmd_llamacache(args: str, _session, settings: dict) -> bool:
    """Disk-persisted prompt-cache slot save/restore for the llamacpp/
    backend (/llamaserve) — EXPERIMENTAL, see caveat below.

    llama-server already does automatic in-memory prompt-cache reuse across
    requests for free (--cache-prompt, on by default) as long as the server
    stays running — no command needed for that; confirmed for real (106x
    lower prefill time on a repeated long system prompt with a real chat
    model).

    This command's save/restore calls a real llama-server API
    (`/slots/{id}?action=save|restore`) and round-trips the right token
    count — but **end-to-end testing across a real server restart does not
    show a cache hit on the next request**, even with the exact same prompt
    and even when explicitly targeting the restored slot via `id_slot`.

    Root cause, confirmed live 2026-08-02 (save → full process restart →
    restore → identical request → `cache_n: 0`, `prompt_ms` unchanged) and
    traced in the vendored `third_party/llama.cpp` source
    (`tools/server/server-context.cpp`, the prompt-reuse block around
    "forcing full prompt re-processing due to lack of cache data"): reusing
    a cached prefix for hybrid/recurrent-memory or SWA models requires an
    in-memory "checkpoint" entry (`slot.prompt.checkpoints`) covering the
    resume position — these checkpoints are built up incrementally DURING
    live generation and are never written by `SLOT_SAVE` or reconstructed by
    `SLOT_RESTORE` (`slot->prompt.clear()` discards them). So even when
    `get_common_prefix()` finds a full token match and the real KV/recurrent
    memory was genuinely restored via `llama_state_seq_load_file`, the empty
    checkpoint list forces `do_reset = true` → full reprocessing, every
    time. This is a known upstream limitation for exactly this class of
    model (the code comments its own reasoning against
    ggml-org/llama.cpp#13194) — it is NOT a greenboost/gb-synapse
    misconfiguration, and the reference workload
    (Qwen3.6-27B-Fable-Fusion, hybrid Gated-DeltaNet architecture) is
    squarely in the affected class. Whether a plain dense-attention model is
    equally affected on this vendored commit is untested. Fixing this for
    real would mean patching the vendored llama.cpp to seed a synthetic
    initial checkpoint on restore — not attempted here. Treat `save`/
    `restore` as not delivering a real speedup for this reference model
    until that upstream gap is closed; only the in-memory reuse above is
    verified to work.

    Usage:
      /llamacache status           — show llama-server cache state
      /llamacache save [key]       — POST /slots/{warm}?action=save
      /llamacache restore [key]    — POST /slots/{warm}?action=restore
      /llamacache erase            — POST /slots/{warm}?action=erase

    key defaults to a hash of the currently configured model. {warm} is
    auto-detected as whichever slot actually holds prompt tokens — slot 0
    is not necessarily the warm one (llama-server load-balances across
    -np slots).
    """
    from greenboost_cli.slash_commands.backend_cmds import (
        _llamacache_slot_action, _llamacache_key, _llamacpp_running_pid,
        _LLAMACPP_SLOT_DIR,
    )

    subcmd = args.strip()
    parts  = subcmd.split(None, 1)
    action = (parts[0].lower() if parts else "status")
    key    = parts[1].strip() if len(parts) > 1 else _llamacache_key(settings)

    if action == "status":
        pid = _llamacpp_running_pid(settings)
        n_files = len(list(_LLAMACPP_SLOT_DIR.glob("*.bin"))) if _LLAMACPP_SLOT_DIR.exists() else 0
        console.print(f"\n  [{GRAY}]gb-synapse:   [/]  {'running' if pid else 'stopped'}")
        console.print(f"  [{GRAY}]Slot dir:     [/]  {_LLAMACPP_SLOT_DIR}")
        console.print(f"  [{GRAY}]Saved slots:  [/]  {n_files}")
        console.print(f"  [{GRAY}]Default key:  [/]  {_llamacache_key(settings)}")
        console.print()
        return True

    if action not in ("save", "restore", "erase"):
        emit_err(f"Unknown /llamacache subcommand: '{action}'  (use: status | save | restore | erase)")
        return True

    if not _llamacpp_running_pid(settings):
        emit_warn("gb-synapse is not running. Start it: /llamaserve")
        return True

    result = _llamacache_slot_action(action, key, settings)
    if result is None:
        emit_err(f"gb-synapse /slots?action={action} failed — is it reachable?")
        return True
    if action == "save":
        emit_ok(f"Saved slot → key '{key}'  ({result.get('n_saved', '?')} tokens)")
    elif action == "restore":
        emit_ok(f"Restored slot ← key '{key}'  ({result.get('n_restored', '?')} tokens) — "
                f"note: not confirmed to produce a cache hit on the next request, see /llamacache help")
    else:
        emit_ok(f"Erased slot cache  ({result.get('n_erased', '?')} tokens)")
    return True


def cmd_gb_status(args: str, _session, _settings) -> bool:
    """Show GreenBoost T1/T2/T3 tier statistics."""
    try:
        from greenboost_cli.greenboost.monitor import get_monitor

        m  = get_monitor()
        s  = m.refresh()

        console.print()
        console.print(f"[bold {VIOLET}]GreenBoost Status[/]")
        console.print(f"[{GRAY}]{'─' * 56}[/]")

        if not s.loaded:
            msg = s.error or "GreenBoost module not loaded"
            emit_warn(msg)
            console.print()
            return True

        loaded_color = LIME if s.loaded else f"dim {GRAY}"
        console.print(f"  [{GRAY}]Module:      [/][{loaded_color}]loaded v{s.version}[/]")
        if s.gpu_name:
            console.print(f"  [{GRAY}]GPU:         [/][{LIME}]{s.gpu_name}[/]")

        # T1 — GPU VRAM
        t1_gb = round(s.vram_physical_mb / 1024, 1)
        console.print(f"\n  [{VIOLET}]T1 VRAM[/]       {t1_gb} GB physical")

        # T2 — DDR pool
        t2_pool  = round(s.ram_pool_mb / 1024, 1)
        t2_alloc = round(s.ram_allocated_mb / 1024, 1)
        t2_avail = round(s.ram_available_mb / 1024, 1)
        t2_pct   = int(s.ram_allocated_mb / s.ram_pool_mb * 100) if s.ram_pool_mb > 0 else 0
        t2_col   = LIME if s.t2_pressure == 0 else (AMBER if s.t2_pressure == 1 else "red")
        console.print(f"  [{VIOLET}]T2 DDR pool[/]    [{t2_col}]{t2_alloc}/{t2_pool} GB used ({t2_pct}%)[/]"
                      f"  avail: {t2_avail} GB  pressure: {s.t2_pressure_label}")

        # T3 — NVMe
        if s.nvme_swap_total_mb > 0:
            t3_total = round(s.nvme_swap_total_mb / 1024, 1)
            t3_used  = round(s.nvme_swap_used_mb / 1024, 1)
            t3_col   = LIME if s.swap_pressure == 0 else (AMBER if s.swap_pressure == 1 else "red")
            console.print(f"  [{VIOLET}]T3 NVMe[/]        [{t3_col}]{t3_used}/{t3_total} GB used[/]"
                          f"  pressure: {s.pressure_label}")

        # Combined
        if s.total_combined_mb > 0:
            console.print(f"\n  [{GRAY}]Total combined:[/]  [{LIME}]{s.total_combined_gb} GB[/]")

        # KV cache
        if s.kv_used_mb > 0 or s.kv_reserve_mb > 0:
            kv_color = AMBER if s.kv_t2_mb > 0 else LIME
            tq_str   = f"  TurboQuant:{s.kv_compression_bits}b" if s.kv_compression_bits > 0 else ""
            console.print(f"  [{GRAY}]KV cache:      [/][{kv_color}]{s.kv_used_mb} MB used"
                          f"  reserve:{s.kv_reserve_mb} MB  T2-spill:{s.kv_t2_mb} MB[/]{tq_str}")

        if s.oom_active:
            emit_warn("OOM recovery is ACTIVE — memory pressure critical!")

        console.print()
        return True

    except Exception as e:
        emit_warn(f"GreenBoost monitor error: {e}")
        return True


def cmd_gb_tiers(args: str, _session, _settings) -> bool:
    """Detailed GreenBoost memory tier breakdown with active buffers."""
    try:
        from greenboost_cli.greenboost.monitor import get_monitor

        m = get_monitor()
        s = m.refresh()

        console.print()
        console.print(f"[bold {VIOLET}]GreenBoost Tier Breakdown[/]")
        console.print(f"[{GRAY}]{'─' * 56}[/]")

        if not s.loaded:
            emit_warn(s.error or "GreenBoost module not loaded")
            console.print()
            return True

        rows = [
            ("T1 VRAM physical",      f"{s.vram_physical_mb:.0f} MB  ({round(s.vram_physical_mb/1024,1)} GB)"),
            ("T2 pool cap",           f"{s.ram_pool_mb:.0f} MB  ({round(s.ram_pool_mb/1024,1)} GB)"),
            ("T2 allocated",          f"{s.ram_allocated_mb:.0f} MB  ({round(s.ram_allocated_mb/1024,1)} GB)"),
            ("T2 available",          f"{s.ram_available_mb:.0f} MB  ({round(s.ram_available_mb/1024,1)} GB)"),
            ("T2 pressure",           s.t2_pressure_label),
            ("T3 NVMe total",         f"{s.nvme_swap_total_mb:.0f} MB"),
            ("T3 NVMe used",          f"{s.nvme_swap_used_mb:.0f} MB"),
            ("T3 GB-alloc",           f"{s.nvme_t3_allocated_mb:.0f} MB"),
            ("T3 pressure",           s.pressure_label),
            ("Active buffers",        str(s.active_buffers)),
            ("KV reserve",            f"{s.kv_reserve_mb} MB"),
            ("KV used",               f"{s.kv_used_mb} MB"),
            ("KV T2 spill",           f"{s.kv_t2_mb} MB"),
            ("TurboQuant bits",       str(s.kv_compression_bits) if s.kv_compression_bits else "off"),
            ("TurboQuant sessions",   str(s.kv_compression_sessions)),
            ("Total combined",        f"{s.total_combined_gb} GB"),
        ]

        for label, value in rows:
            console.print(f"  [{GRAY}]{label:<22}[/]  [{LIME}]{value}[/]")

        console.print()
        return True

    except Exception as e:
        emit_warn(f"GreenBoost monitor error: {e}")
        return True


def _get_inference_status(settings: dict) -> dict:
    """Probe gb-synapse's llama-server and GPU processes."""
    result: dict = {
        "server_running": False,
        "server_url": "",
        "server_models": [],
        "server_error": "",
        "gpu_processes": [],
        "gpu_stats": {},
        "configured_model": settings.get("model", ""),
    }

    from greenboost_cli.slash_commands.backend_cmds import _llamacpp_base_url
    result["server_url"] = _llamacpp_base_url(settings)

    # Probe server
    if base_url:
        models_url = base_url.rstrip("/") + "/models"
        try:
            req = urllib.request.Request(
                models_url,
                headers={"User-Agent": "GreenBoostCLI/1.0", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(req, timeout=2) as r:
                import json
                data = json.loads(r.read())
                result["server_running"] = True
                result["server_models"] = [m["id"] for m in data.get("data", [])]
        except Exception as e:
            result["server_error"] = str(e)

    # GPU hardware stats via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,power.draw,power.limit,"
             "utilization.gpu,utilization.memory,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        if out:
            parts = [p.strip() for p in out.split(",")]
            result["gpu_stats"] = {
                "name":       parts[0] if len(parts) > 0 else "",
                "temp_c":     parts[1] if len(parts) > 1 else "?",
                "power_w":    parts[2] if len(parts) > 2 else "?",
                "power_lim":  parts[3] if len(parts) > 3 else "?",
                "gpu_util":   parts[4] if len(parts) > 4 else "?",
                "mem_util":   parts[5] if len(parts) > 5 else "?",
                "mem_used":   parts[6] if len(parts) > 6 else "?",
                "mem_free":   parts[7] if len(parts) > 7 else "?",
                "mem_total":  parts[8] if len(parts) > 8 else "?",
            }
    except Exception:
        pass

    # GPU compute processes
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid = parts[0]
                name = Path(parts[1]).name
                mem  = parts[2]
                # Try to get cmdline for better identification
                cmdline = ""
                try:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()[:80]
                except Exception:
                    pass
                result["gpu_processes"].append({
                    "pid": pid, "name": name, "mem_mib": mem, "cmdline": cmdline
                })
    except Exception:
        pass

    return result


def _render_inference_section(inf: dict) -> None:
    """Render the inference server diagnostics block."""
    console.print(f"\n[bold {VIOLET}]Inference Server[/]")

    model = inf["configured_model"]
    console.print(f"  [{GRAY}]Configured model:[/]  [{VIOLET}]{model}[/]")

    # Server connectivity
    url = inf["server_url"]
    if url:
        if inf["server_running"]:
            console.print(f"  [{GRAY}]Server:[/]  [{LIME}]UP[/]  [{GRAY}]{url}[/]")
            for m in inf["server_models"]:
                console.print(f"    [{LIME}]◈[/]  [{GRAY}]{m}[/]")
        else:
            console.print(f"  [{GRAY}]Server:[/]  [red]DOWN[/]  [{GRAY}]{url}[/]")
            err = inf["server_error"]
            if "refused" in err.lower() or "111" in err:
                console.print(f"  [dim {GRAY}]Start it:[/]  [{LIME}]/llamaserve {model}[/]")
            else:
                console.print(f"  [dim {GRAY}]Error: {err[:120]}[/]")
    else:
        console.print(f"  [{GRAY}]Server URL not configured[/]")

    # GPU stats
    gs = inf.get("gpu_stats", {})
    if gs:
        util   = gs.get("gpu_util", "?")
        mem_u  = gs.get("mem_used", "?")
        mem_t  = gs.get("mem_total", "?")
        temp   = gs.get("temp_c", "?")
        pwr    = gs.get("power_w", "?")
        plim   = gs.get("power_lim", "?")
        util_i = int(util) if str(util).isdigit() else 0
        u_col  = LIME if util_i < 50 else (AMBER if util_i < 85 else "red")
        console.print(
            f"\n  [{GRAY}]GPU:[/]  [{LIME}]{gs.get('name', '?')}[/]"
            f"  [{GRAY}]temp[/] [{LIME}]{temp}°C[/]"
            f"  [{GRAY}]power[/] [{LIME}]{pwr}/{plim}W[/]"
        )
        console.print(
            f"  [{GRAY}]Util:[/]  [{u_col}]{util}%[/]"
            f"  [{GRAY}]VRAM:[/]  [{LIME}]{mem_u}/{mem_t} MiB[/]"
        )

    # GPU compute processes
    procs = inf.get("gpu_processes", [])
    if procs:
        console.print(f"\n  [{GRAY}]GPU compute processes:[/]")
        for p in procs:
            is_llama = "llama-server" in p["cmdline"].lower() or "llama-server" in p["name"].lower()
            row_col = LIME if is_llama else GRAY
            console.print(
                f"    [{row_col}]pid {p['pid']:>7}[/]  [{LIME}]{p['mem_mib']:>6} MiB[/]"
                f"  [{GRAY}]{p['name']}[/]"
            )
            if p["cmdline"] and is_llama:
                console.print(f"    [dim {GRAY}]  {p['cmdline'][:100]}[/]")
    else:
        console.print(f"\n  [dim {GRAY}]No GPU compute processes (model not loaded)[/]")


def _render_full_vitals(s, settings: dict | None = None) -> None:
    """Render comprehensive debug vitals using Rich console."""
    from rich.table import Table

    console.print()
    dv_state = f"[{LIME}]ON[/]" if s.debug_vitals_enabled else f"[dim {GRAY}]OFF[/]"
    console.print(f"[bold {VIOLET}]{'═' * 60}[/]")
    console.print(f"[bold {VIOLET}]GreenBoost Debug Vitals[/]  [dim]v{s.version} · debug_vitals: {dv_state}[/]")
    console.print(f"[dim {GRAY}]{'─' * 60}[/]")

    # Inference server section (most actionable info first)
    if settings is not None:
        inf = _get_inference_status(settings)
        _render_inference_section(inf)

    # Memory tiers
    console.print(f"\n[bold {VIOLET}]Memory Tiers[/]")
    t1_pct = int(s.vram_physical_mb / (s.vram_physical_mb or 1) * 100) if s.vram_physical_mb else 0
    t2_pct = int(s.ram_allocated_mb / (s.ram_pool_mb or 1) * 100) if s.ram_pool_mb else 0
    t3_pct = int(s.nvme_swap_used_mb / (s.nvme_swap_total_mb or 1) * 100) if s.nvme_swap_total_mb else 0
    t2_col = LIME if s.t2_pressure == 0 else (AMBER if s.t2_pressure == 1 else "red")
    t3_col = LIME if s.swap_pressure == 0 else (AMBER if s.swap_pressure == 1 else "red")
    console.print(f"  [{GRAY}]T1 VRAM[/]       [{LIME}]{s.vram_physical_mb:.0f} MB ({round(s.vram_physical_mb/1024,1)} GB)[/]")
    console.print(f"  [{GRAY}]T2 DDR pool[/]   [{t2_col}]{s.ram_allocated_mb:.0f} / {s.ram_pool_mb:.0f} MB[/]"
                  f"  [{t2_col}]{t2_pct}%  {s.t2_pressure_label}[/]")
    console.print(f"  [{GRAY}]T3 NVMe[/]       [{t3_col}]{s.nvme_swap_used_mb:.0f} / {s.nvme_swap_total_mb:.0f} MB[/]"
                  f"  [{t3_col}]{t3_pct}%  {s.pressure_label}[/]")
    console.print(f"  [{GRAY}]Combined:[/]     [{LIME}]{s.total_combined_gb} GB[/]"
                  f"  [{GRAY}]Active DMA-BUF:[/]  [{LIME}]{s.active_buffers}[/]")
    if s.nvme_swap_used_mb > 0:
        console.print(f"  [bold red]WARNING: T3 spillover active — ~100x slowdown![/]")

    # KV cache
    console.print(f"\n[bold {VIOLET}]KV Cache[/]")
    kv_col = AMBER if s.kv_t2_mb > 0 else LIME
    tq_str = f"[{LIME}]{s.kv_compression_bits}b  sessions:{s.kv_compression_sessions}[/]" \
             if s.kv_compression_bits > 0 else f"[dim {GRAY}]off[/]"
    console.print(f"  [{GRAY}]KV used:[/]      [{kv_col}]{s.kv_used_mb} MB[/]"
                  f"  [{GRAY}]reserve:[/]  [{LIME}]{s.kv_reserve_mb} MB[/]"
                  f"  [{GRAY}]T2 spill:[/]  [{kv_col}]{s.kv_t2_mb} MB[/]")
    console.print(f"  [{GRAY}]TurboQuant:[/]   {tq_str}")

    # Shim stats
    console.print(f"\n[bold {VIOLET}]Shim Stats[/]")
    if not s.shim_stale:
        phase_col = LIME if s.shim_phase in ("INFERENCE", "STEADY") else (
            AMBER if s.shim_phase == "MODEL_LOAD" else "red")
        path_col = LIME if s.shim_active_path in ("A", "A0") else (
            AMBER if s.shim_active_path == "B" else "red")
        console.print(f"  [{GRAY}]Phase:[/]   [{phase_col}]{s.shim_phase or '—'}[/]"
                      f"  [{GRAY}]Path:[/]  [{path_col}]{s.shim_active_path or '?'}[/]"
                      f"  [{GRAY}]Stale:[/]  [{LIME}]no[/]")
        tbl = Table(box=None, padding=(0, 2), show_header=False)
        tbl.add_column(style=GRAY); tbl.add_column(style=LIME)
        tbl.add_column(style=GRAY); tbl.add_column(style=LIME)
        tbl.add_row("Path A0:",   str(s.shim_path_a0),   "Path A:",  str(s.shim_path_a))
        tbl.add_row("Path B:",    str(s.shim_path_b),    "Path C:",  str(s.shim_path_c))
        tbl.add_row("H2D:",       f"{s.shim_h2d_mb} MB", "D2H:",     f"{s.shim_d2h_mb} MB")
        tbl.add_row("Dispatch:",  str(s.shim_kernel_dispatch), "Headroom:", f"{s.shim_headroom_mb} MB")
        tbl.add_row("T2 frag:",   f"{s.shim_frag_pct}%", "Cold evicts:", str(s.shim_cold_evicts))
        tbl.add_row("KV dedup:",  str(s.shim_kv_dedup),  "KV frag:", f"{s.shim_kv_frag_mb} MB")
        tbl.add_row("Rem allocs:", str(s.shim_remote_alloc_count),
                    "Rem MB:",   str(s.shim_remote_alloc_mb))
        console.print(tbl)
    else:
        console.print(f"  [dim {GRAY}]Shim stats unavailable — no inference process active.[/]")
        console.print(f"  [dim {GRAY}]Start with: GREENBOOST_ACTIVE=1 LD_PRELOAD=libgreenboost_cuda.so <app>[/]")

    # GPU extended
    console.print(f"\n[bold {VIOLET}]GPU Hardware[/]")
    if s.gpu_name:
        temp_col = LIME if s.gpu_temp_c < 80 else (AMBER if s.gpu_temp_c < 90 else "red")
        console.print(f"  [{GRAY}]GPU:[/]  [{LIME}]{s.gpu_name}[/]")
        console.print(f"  [{GRAY}]Temp:[/]  [{temp_col}]{s.gpu_temp_c}°C[/]"
                      f"  [{GRAY}]Power:[/]  [{LIME}]{s.gpu_power_w:.0f}/{s.gpu_power_limit_w:.0f}W[/]"
                      f"  [{GRAY}]GPU:[/]  [{LIME}]{s.gpu_util_pct}%[/]"
                      f"  [{GRAY}]Mem:[/]  [{LIME}]{s.gpu_mem_util_pct}%[/]")
        console.print(f"  [{GRAY}]SM:[/]   [{LIME}]{s.gpu_sm_clock_mhz} MHz[/]"
                      f"  [{GRAY}]Mem clock:[/]  [{LIME}]{s.gpu_mem_clock_mhz} MHz[/]")
    else:
        console.print(f"  [dim {GRAY}]nvidia-smi not available[/]")

    # NVTX tail
    console.print(f"\n[bold {VIOLET}]Recent NVTX Events (last 10)[/]")
    if s.nvtx_tail:
        for line in s.nvtx_tail[-10:]:
            parts = line.split(None, 4)
            ev_col = GRAY
            if len(parts) >= 2:
                ev = parts[1]
                if ev.startswith("ALLOC_T1"): ev_col = LIME
                elif ev.startswith("ALLOC_T2"): ev_col = "cyan"
                elif ev.startswith("ALLOC_T3"): ev_col = AMBER
                elif ev.startswith("PHASE_"):   ev_col = VIOLET
                elif ev.startswith("OOM_"):     ev_col = "red"
                elif ev.startswith("FEEDER_"):  ev_col = "magenta"
            console.print(f"  [dim]{parts[0][:16] if parts else '?'}[/]  "
                          f"[{ev_col}]{parts[1] if len(parts) > 1 else '?':<22}[/]  "
                          f"[dim {GRAY}]{' '.join(parts[2:]) if len(parts) > 2 else ''}[/]")
    else:
        console.print(f"  [dim {GRAY}]No events — log empty or shim inactive. Live: greenboost nvtx-logs[/]")

    # AppArmor denials
    import re as _re
    console.print(f"\n[bold {VIOLET}]AppArmor[/]")
    aa_count = 0
    try:
        out = subprocess.check_output(
            ["journalctl", "-k", "--grep=apparmor.*greenboost", "--no-pager", "-n", "30",
             "--output=short"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode(errors="replace").strip()
        aa_lines = [l for l in out.splitlines() if l.strip() and not l.startswith("--")]
        aa_count = len(aa_lines)
        if aa_count == 0:
            console.print(f"  [{LIME}]✓ No AppArmor denials[/]")
        else:
            console.print(f"  [red]✗ {aa_count} denial(s) in kernel log[/]")
            for ln in aa_lines[-3:]:
                m_op   = _re.search(r'operation="([^"]+)"', ln)
                m_name = _re.search(r'name="([^"]+)"', ln)
                op   = m_op.group(1)   if m_op   else ""
                name = m_name.group(1).split("/")[-1] if m_name else ""
                console.print(f"    [dim {GRAY}]DENIED {op}  {name}[/]")
            console.print(f"  [dim {GRAY}]Fix: sudo greenboost install-sys-configs[/]")
    except Exception:
        console.print(f"  [dim {GRAY}]journalctl unavailable[/]")

    # gb-synapse / shim injection status
    console.print(f"\n[bold {VIOLET}]Shim Injection[/]")
    shim_path = Path("/usr/local/lib/libgreenboost_cuda.so")
    if shim_path.exists():
        console.print(f"  [{LIME}]✓ Shim present:[/] [{GRAY}]{shim_path}[/]")
    else:
        console.print(f"  [red]✗ Shim NOT installed[/]  [{GRAY}]Build + install first[/]")
    ld_preload  = os.environ.get("LD_PRELOAD", "")
    shim_in_ld  = str(shim_path) in ld_preload
    if shim_in_ld:
        console.print(f"  [{LIME}]✓ LD_PRELOAD: shim active in this process[/]")
    else:
        console.print(f"  [dim {GRAY}]LD_PRELOAD: shim not in current env (OK for CLI)[/]")
    # Check gb-synapse's llama-server process for shim injection
    try:
        llama_pids = subprocess.check_output(
            ["pgrep", "-f", "llama-server"], text=True, stderr=subprocess.DEVNULL
        ).strip().split()
        for pid in llama_pids[:3]:
            maps = Path(f"/proc/{pid}/maps")
            has_shim = False
            try:
                has_shim = str(shim_path) in maps.read_text(errors="replace")
            except Exception:
                pass
            shim_icon = f"[{LIME}]✓[/]" if has_shim else "[red]✗[/]"
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()[:60]
            except Exception:
                cmd = f"pid {pid}"
            console.print(f"  gb-synapse pid [{GRAY}]{pid}[/]  shim:{shim_icon}  [{GRAY}]{cmd}[/]")
    except Exception:
        pass

    # Diffuser vitals
    console.print(f"\n[bold {VIOLET}]Diffuser Vitals[/]")
    _vitals_paths = [Path("/run/greenboost/diffuser_vitals.json")]
    try:
        from greenboost_cli.environment.settings import GB_HOME as _GB_HOME
        _vitals_paths.append(_GB_HOME / "diffuser_vitals.json")
    except Exception:
        pass
    diff_vitals = None
    for _vp in _vitals_paths:
        if _vp.exists():
            try:
                import json as _json
                diff_vitals = _json.loads(_vp.read_text())
                break
            except Exception:
                pass
    if diff_vitals:
        import time as _time
        age = int(_time.time() - diff_vitals.get("ts", 0))
        stale = age > 120
        state = diff_vitals.get("state", "unknown")
        model = diff_vitals.get("model", "?")
        pipeline = diff_vitals.get("pipeline", "?")
        vram_a = diff_vitals.get("vram_alloc_mb", 0)
        vram_p = diff_vitals.get("vram_peak_mb", 0)
        t2_mb  = diff_vitals.get("t2_alloc_mb", 0)
        gen_s  = diff_vitals.get("last_gen_s", 0)
        step   = diff_vitals.get("gen_step", 0)
        total  = diff_vitals.get("gen_total_steps", 0)
        prompt = diff_vitals.get("last_prompt", "")
        age_str = f"{age}s ago" if not stale else f"[dim]{age}s ago (stale)[/]"
        state_col = LIME if state == "ready" else (VIOLET if state == "generating" else (AMBER if state == "loading" else GRAY))
        console.print(f"  [{GRAY}]Pipeline:[/]  [{LIME}]{pipeline}[/]  [{GRAY}]model:[/]  [{VIOLET}]{model}[/]  [{GRAY}]updated:[/]  {age_str}")
        console.print(f"  [{GRAY}]State:[/]     [{state_col}]{state}[/]"
                      f"  [{GRAY}]VRAM:[/]  [{LIME}]{vram_a}/{vram_p} MB (alloc/peak)[/]"
                      + (f"  [{AMBER}]T2: {t2_mb} MB[/]" if t2_mb > 0 else ""))
        if gen_s:
            console.print(f"  [{GRAY}]Last gen:[/]  [{LIME}]{gen_s:.1f}s[/]"
                          + (f"  [{GRAY}]steps:[/] {step}/{total}" if total else ""))
        if prompt:
            console.print(f"  [{GRAY}]Prompt:[/]    [dim]{prompt[:80]}[/]")
    else:
        console.print(f"  [dim {GRAY}]No active diffuser pipeline (start with /design or gen_art.py)[/]")

    # Footer
    console.print(f"\n[dim {GRAY}]{'─' * 60}[/]")
    console.print(f"  [dim]Tip: /gb-vitals on|off  ·  /gb-status  ·  /gb-tiers  ·  /gb-pool-cap[/]")
    if s.debug_vitals_enabled:
        console.print(f"  [{LIME}]debug vitals: ON[/]  [dim]disable: sudo greenboost debug vitals off[/]")
    else:
        console.print(f"  [dim {GRAY}]debug vitals: OFF  enable: sudo greenboost debug vitals on[/]")
    console.print(f"[bold {VIOLET}]{'═' * 60}[/]")
    console.print()


def cmd_gb_vitals(args: str, _session, settings) -> bool:
    """Full GreenBoost debug vitals — /gb-vitals [on|off|--export <file>]"""
    parts = args.strip().split()
    subcmd = (parts[0].lower() if parts else "") or "show"

    if subcmd in ("on", "off"):
        if not _gb_installed():
            emit_warn("GreenBoost is not installed.")
            return True
        result = subprocess.run(["sudo", "greenboost", "debug", "vitals", subcmd])
        if result.returncode == 0:
            state = "ENABLED" if subcmd == "on" else "DISABLED"
            emit_ok(f"Debug vitals {state}")
        else:
            emit_err("Could not toggle debug vitals (sudo required)")
        return True

    export_path: "str | None" = None
    if subcmd == "--export" and len(parts) >= 2:
        export_path = parts[1]
    elif len(parts) >= 2 and parts[0] == "--export":
        export_path = parts[1]

    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        m = get_monitor()
        s = m.refresh()
        if export_path:
            _export_vitals_md(s, settings, export_path)
        else:
            _render_full_vitals(s, settings)
    except Exception as e:
        emit_warn(f"GreenBoost vitals error: {e}")
        if settings:
            inf = _get_inference_status(settings)
            _render_inference_section(inf)
    return True


def _export_vitals_md(s, settings: "dict | None", path: str) -> None:
    """Export full vitals as a Markdown snapshot to *path*."""
    import time as _time
    lines = []
    lines.append("# GreenBoost Vitals Snapshot")
    lines.append(f"\n**Generated:** {_time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Module:** v{s.version}  **GPU:** {s.gpu_name or '—'}\n")

    lines.append("## Memory Tiers")
    lines.append(f"| Tier | Used | Total | % |")
    lines.append(f"|------|------|-------|---|")
    t1gb = round(s.vram_physical_mb / 1024, 1) if s.vram_physical_mb else 0
    t2u  = round(s.ram_allocated_mb / 1024, 1)
    t2t  = round(s.ram_pool_mb / 1024, 1)
    t2p  = int(t2u / t2t * 100) if t2t else 0
    t3u  = round(s.nvme_swap_used_mb / 1024, 1)
    t3t  = round(s.nvme_swap_total_mb / 1024, 1)
    t3p  = int(t3u / t3t * 100) if t3t else 0
    lines.append(f"| T1 GPU VRAM | {t1gb} GB | {t1gb} GB | 100% |")
    lines.append(f"| T2 DDR Pool | {t2u} GB | {t2t} GB | {t2p}% |")
    lines.append(f"| T3 NVMe     | {t3u} GB | {t3t} GB | {t3p}% |")
    lines.append(f"| **Combined** | | **{s.total_combined_gb} GB** | |")
    if s.nvme_swap_used_mb > 0:
        lines.append(f"\n> ⚠ T3 spillover active — ~100× slowdown!")

    lines.append("\n## KV Cache")
    lines.append(f"- Used: {s.kv_used_mb} MB")
    lines.append(f"- Reserve: {s.kv_reserve_mb} MB")
    lines.append(f"- T2 spill: {s.kv_t2_mb} MB")
    tq = f"{s.kv_compression_bits}-bit ({s.kv_compression_sessions} sessions)" if s.kv_compression_bits else "off"
    lines.append(f"- TurboQuant: {tq}")
    if s._kv_t3_stale:
        lines.append(f"- KV T3: 0 MB *(kernel counter stale — T3 allocation is 0)*")

    lines.append("\n## Shim Stats")
    if not s.shim_stale:
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Phase | {s.shim_phase} |")
        lines.append(f"| Active path | {s.shim_active_path} |")
        lines.append(f"| Path B allocs | {s.shim_path_b} |")
        lines.append(f"| H2D | {s.shim_h2d_mb} MB |")
        lines.append(f"| D2H | {s.shim_d2h_mb} MB |")
        lines.append(f"| Headroom | {s.shim_headroom_mb} MB |")
        lines.append(f"| T2 frag | {s.shim_frag_pct}% |")
        lines.append(f"| Cold evictions | {s.shim_cold_evicts} |")
        lines.append(f"| KV dedup hits | {s.shim_kv_dedup} |")
        lines.append(f"| Kernel dispatches | {s.shim_kernel_dispatch} |")
    else:
        lines.append("*Shim stats unavailable (no active inference process)*")

    lines.append("\n## GPU Hardware")
    if s.gpu_name:
        lines.append(f"- GPU: {s.gpu_name}")
        lines.append(f"- Temperature: {s.gpu_temp_c}°C")
        lines.append(f"- Power: {s.gpu_power_w:.0f}/{s.gpu_power_limit_w:.0f} W")
        lines.append(f"- GPU utilization: {s.gpu_util_pct}%")
        lines.append(f"- Memory utilization: {s.gpu_mem_util_pct}%")
        lines.append(f"- SM clock: {s.gpu_sm_clock_mhz} MHz")
        lines.append(f"- Mem clock: {s.gpu_mem_clock_mhz} MHz")

    lines.append("\n## Recent NVTX Events")
    if s.nvtx_tail:
        lines.append("```")
        lines.extend(s.nvtx_tail[-20:])
        lines.append("```")
    else:
        lines.append("*No events*")

    lines.append("\n## Inference Server")
    if settings:
        try:
            inf = _get_inference_status(settings)
            lines.append(f"- Model: {inf.get('configured_model', '—')}")
            lines.append(f"- Server: {'UP' if inf.get('server_running') else 'DOWN'}  {inf.get('server_url', '—')}")
            gs = inf.get("gpu_stats", {})
            if gs:
                lines.append(f"- GPU temp: {gs.get('temp_c', '?')}°C  power: {gs.get('power_w', '?')}W")
        except Exception:
            pass

    out = "\n".join(lines) + "\n"
    Path(path).write_text(out)
    emit_ok(f"Vitals exported → {path}")


def cmd_gb_kv_reserve(args: str, _session, _settings) -> bool:
    """Set GreenBoost KV cache T1 reserve. Usage: /gb-kv-reserve <mb>"""
    arg = args.strip()
    if not arg:
        # Show current value
        try:
            from greenboost_cli.greenboost.monitor import get_monitor
            s = get_monitor().refresh()
            console.print(f"\n  [{GRAY}]KV reserve:[/]  [{LIME}]{s.kv_reserve_mb} MB[/]"
                          f"  [{GRAY}]used:[/]  [{LIME}]{s.kv_used_mb} MB[/]"
                          f"  [{GRAY}]T2 spill:[/]  [{LIME}]{s.kv_t2_mb} MB[/]\n")
        except Exception as e:
            emit_warn(f"Could not read KV reserve: {e}")
        return True
    try:
        mb = int(arg)
    except ValueError:
        emit_err(f"Expected integer MB, got: {arg!r}")
        return True
    if mb < 0:
        emit_err("Reserve must be ≥ 0 MB")
        return True
    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        ok = get_monitor().set_kv_reserve(mb)
        if ok:
            emit_ok(f"KV reserve set to {mb} MB")
        else:
            emit_warn("IOCTL failed — GreenBoost device not accessible")
    except Exception as e:
        emit_warn(f"set_kv_reserve error: {e}")
    return True


def cmd_gb_pool_cap(args: str, _session, _settings) -> bool:
    """Set or auto-compute the GreenBoost T2 DDR pool cap.

    Usage:
      /gb-pool-cap          — show current cap
      /gb-pool-cap auto     — compute optimal cap from available RAM
      /gb-pool-cap <gb>     — set cap to <gb> GB (e.g. /gb-pool-cap 24)
    """
    from greenboost_cli.greenboost.monitor import get_monitor, GreenBoostMonitor
    arg = args.strip().lower()

    if not arg:
        try:
            s = get_monitor().refresh()
            if not s.loaded:
                emit_warn("GreenBoost not loaded")
                return True
            pool_gb  = round(s.ram_pool_mb / 1024, 1)
            alloc_gb = round(s.ram_allocated_mb / 1024, 1)
            avail_gb = round(s.ram_available_mb / 1024, 1)
            console.print(f"\n  [{GRAY}]T2 pool cap:[/]  [{LIME}]{pool_gb} GB[/]"
                          f"  [{GRAY}]allocated:[/] [{LIME}]{alloc_gb} GB[/]"
                          f"  [{GRAY}]available:[/] [{LIME}]{avail_gb} GB[/]\n")
        except Exception as e:
            emit_warn(f"Could not read pool cap: {e}")
        return True

    if arg == "auto":
        try:
            m = get_monitor()
            ok, cap_mb = m.apply_dynamic_pool_cap(safety_reserve_gb=9, target_pct=0.80)
            if ok:
                emit_ok(f"T2 pool cap auto-set to {round(cap_mb / 1024, 1)} GB")
            else:
                emit_warn("IOCTL failed — GreenBoost device not accessible")
        except Exception as e:
            emit_warn(f"Pool cap error: {e}")
        return True

    try:
        cap_gb = float(arg)
    except ValueError:
        emit_err(f"Expected 'auto' or GB value, got: {arg!r}")
        return True
    cap_mb = int(cap_gb * 1024)
    try:
        ok, actual_mb = get_monitor().set_pool_cap(cap_mb)
        if ok:
            emit_ok(f"T2 pool cap set to {round(actual_mb / 1024, 1)} GB")
        else:
            emit_warn("IOCTL failed — GreenBoost device not accessible")
    except Exception as e:
        emit_warn(f"set_pool_cap error: {e}")
    return True


def cmd_clear_memory(args: str, _session, _settings) -> bool:
    """Run `sudo greenboost clear memory-pool` to release T1+T2 memory."""
    import subprocess
    emit_info("Clearing GreenBoost memory pool (T1 + T2)…")
    result = subprocess.run(
        ["sudo", "greenboost", "clear", "memory-pool"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        emit_ok("Memory pool cleared.")
    else:
        msg = (result.stderr or result.stdout or "unknown error").strip()
        emit_warn(f"clear memory-pool: {msg}")
    return True


def _gb_py_call(fn: str, *args: str) -> "tuple[bool, str]":
    """Call a gb_synapse function through the installed greenboost python root.

    Goes through gb_paths rather than importing gb_synapse into this venv:
    greenboost-cli ships its own environment and the orchestrator layer lives
    outside it (see greenboost_cli/gb_paths.py, the same resolution every other
    bridge here uses)."""
    import json as _json
    import subprocess
    from greenboost_cli import gb_paths
    root = gb_paths.py_root()
    code = (f"import sys, json; sys.path.insert(0, {str(root)!r}); "
            f"import gb_synapse; print(json.dumps(gb_synapse.{fn}(*{list(args)!r})))")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "unknown error").strip()
    return True, r.stdout.strip()


def cmd_pause(args: str, _session, settings: dict) -> bool:
    """Pause the local model: save its KV state, stop the engine, free the VRAM.

    The problem it solves, in the operator's terms: an idle model keeps most of
    the card allocated while doing nothing (measured on this box, 10.4 GiB of
    12 GB at 0% GPU usage). Pausing gives that back without losing the
    conversation , the KV cache is written to disk and restored on /resume.
    """
    import json as _json
    model = args.strip() or settings.get("model", "")
    if not model:
        emit_warn("No model to pause. Usage: /pause [model]")
        return True
    force = False
    if model.endswith(" --force"):
        model, force = model[:-8].strip(), True

    emit_info(f"Pausing {model} , saving KV state, then releasing the GPU…")
    ok, out = _gb_py_call("pause", model, *(["--force"] if force else []))
    if not ok:
        emit_err(f"pause failed: {out}")
        return True
    try:
        d = _json.loads(out)
    except Exception:
        emit_warn(out); return True

    if not d.get("ok"):
        err = d.get("error", "unknown")
        emit_warn(err)
        if d.get("busy_slots"):
            emit_info("Nothing was saved and the model is still serving. "
                      "Re-run once the reply finishes, or `/pause --force` to "
                      "accept losing the reply in progress.")
        return True

    freed = d.get("vram_freed_mb")
    tok = d.get("tokens_saved", 0)
    emit_ok(f"Paused. {tok:,} tokens of context saved"
            + (f", {freed} MB of VRAM returned." if freed else "."))
    emit_info("The conversation is intact , `/resume` brings it back.")
    return True


def cmd_resume(args: str, _session, settings: dict) -> bool:
    """Resume a paused model and restore its saved KV state."""
    import json as _json
    model = args.strip() or settings.get("model", "")
    if not model:
        emit_warn("No model to resume. Usage: /resume [model]")
        return True

    emit_info(f"Resuming {model} , reloading weights, then restoring context…")
    ok, out = _gb_py_call("resume", model)
    if not ok:
        emit_err(f"resume failed: {out}")
        return True
    try:
        d = _json.loads(out)
    except Exception:
        emit_warn(out); return True

    if not d.get("ok") and not d.get("slots_restored"):
        emit_warn(d.get("error", "resume failed"))
        return True

    tok = d.get("tokens_restored", 0)
    emit_ok(f"Resumed. {tok:,} tokens of context restored "
            f"(weights {d.get('reload_s', '?')}s, context {d.get('restore_s', '?')}s).")
    if d.get("errors"):
        emit_warn("The model is serving again, but its cached context could not "
                  "be restored, so the next reply will re-read the conversation "
                  "from scratch. Nothing was lost, the first reply is just slower.")
    return True


def cmd_paused(args: str, _session, _settings) -> bool:
    """List paused sessions waiting to be resumed."""
    import json as _json
    ok, out = _gb_py_call("paused")
    if not ok:
        emit_err(out); return True
    try:
        rows = _json.loads(out)
    except Exception:
        emit_warn(out); return True
    if not rows:
        emit_info("No paused sessions.")
        return True
    for d in rows:
        tok = sum(int(s.get("tokens") or 0) for s in d.get("slots", []))
        gb = (d.get("disk_bytes") or 0) / (1024 ** 3)
        console.print(f"  [{VIOLET}]{d.get('model')}[/]  "
                      f"[{GRAY}]{tok:,} tokens · {gb:.2f} GB on disk · "
                      f"paused {d.get('idle_s', 0):.0f}s ago[/]")
    return True


def register(command_table: dict) -> None:
    """Register GreenBoost commands into the command table."""
    command_table["turboquant"]     = cmd_turboquant
    command_table["llamacache"]     = cmd_llamacache
    command_table["gb-status"]      = cmd_gb_status
    command_table["gb-tiers"]       = cmd_gb_tiers
    command_table["gb-vitals"]      = cmd_gb_vitals
    command_table["gb-kv-reserve"]  = cmd_gb_kv_reserve
    command_table["gb-pool-cap"]    = cmd_gb_pool_cap
    command_table["clear-memory"]   = cmd_clear_memory
    command_table["pause"]          = cmd_pause
    command_table["resume"]         = cmd_resume
    command_table["paused"]         = cmd_paused
