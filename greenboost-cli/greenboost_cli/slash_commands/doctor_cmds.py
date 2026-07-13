"""Health diagnostics for GreenBoost CLI — /doctor command.

Runs a suite of modular checks across the full stack and prints a grouped
status report.  Exit behaviour mirrors doctor.py from optimal-claude:
  - soft=False failures → hard error (required component broken)
  - soft=True  failures → warnings only (optional feature degraded)

Usage:
  /doctor            Full diagnostic run
  /doctor --fix      Print fix commands only for failing checks
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from greenboost_cli.gb_paths import gb_py_root, gb_root_hint
from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, VIOLET, GRAY, LIME, AMBER, TEAL, DIM, CYAN, RED,
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    soft: bool = True   # soft=True → warning only; soft=False → hard error


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 10)
    return Check(
        name="Python ≥ 3.10",
        ok=ok,
        detail=f"{v.major}.{v.minor}.{v.micro}",
        fix="Install Python 3.10+",
        soft=False,
    )


def _check_torch_cuda() -> Check:
    try:
        import torch
        if not torch.cuda.is_available():
            return Check(
                name="torch + CUDA",
                ok=False,
                detail=f"torch {torch.__version__} — CUDA unavailable",
                fix="pip install torch --index-url https://download.pytorch.org/whl/cu130",
            )
        cuda_v = torch.version.cuda or "?"
        dev    = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        return Check(
            name="torch + CUDA",
            ok=True,
            detail=f"torch {torch.__version__} · CUDA {cuda_v} · {dev} · {vram} GB",
        )
    except ImportError:
        return Check(
            name="torch + CUDA",
            ok=False,
            detail="not installed — RAG / diffusion / inference unavailable",
            fix="pip install greenboost-cli[rag]",
        )


def _check_sentence_transformers() -> Check:
    try:
        import sentence_transformers as st
        return Check(name="sentence-transformers", ok=True, detail=f"v{st.__version__}")
    except ImportError:
        return Check(
            name="sentence-transformers",
            ok=False,
            detail="not installed — RAG embedding unavailable",
            fix="pip install greenboost-cli[rag]",
        )


def _check_rag_index() -> Check:
    try:
        from greenboost_cli.rag.engine import _load_store, _load_folders
        _, meta  = _load_store()
        folders  = _load_folders()
        n        = len(meta) if meta else 0
        if n == 0:
            return Check(
                name="RAG index",
                ok=False,
                detail="empty — no documents indexed",
                fix="/rag-add <folder>  to index a project",
            )
        return Check(
            name="RAG index",
            ok=True,
            detail=f"{n:,} chunks · {len(folders)} source(s)",
        )
    except Exception as e:
        return Check(
            name="RAG index",
            ok=False,
            detail=f"load error: {e}",
            fix="pip install greenboost-cli[rag]",
        )


def _check_openai_sdk() -> Check:
    try:
        import openai
        return Check(
            name="openai SDK",
            ok=True,
            detail=f"v{openai.__version__}",
        )
    except ImportError:
        return Check(
            name="openai SDK",
            ok=False,
            detail="not installed — gb-synapse (OpenAI-compatible) client degraded",
            fix="pip install openai",
        )


def _check_ollama() -> Check:
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            models = [l for l in r.stdout.strip().splitlines() if l and "NAME" not in l]
            return Check(
                name="ollama",
                ok=True,
                detail=f"running · {len(models)} model(s)",
            )
        return Check(
            name="ollama",
            ok=False,
            detail="ollama list failed — daemon not running?",
            fix="ollama serve  (or: systemctl start ollama)",
        )
    except FileNotFoundError:
        return Check(
            name="ollama",
            ok=False,
            detail="not installed",
            fix="curl -fsSL https://ollama.com/install.sh | sh",
        )
    except Exception as e:
        return Check(name="ollama", ok=False, detail=f"error: {e}")


def _check_mcp_module() -> Check:
    try:
        from greenboost_cli.mcp import server as _ms  # noqa: F401
        return Check(
            name="MCP server",
            ok=True,
            detail="importable — /mcp config for setup snippet",
        )
    except ImportError as e:
        return Check(
            name="MCP server",
            ok=False,
            detail=f"not available: {e}",
            fix="pip install greenboost-cli[mcp]",
        )


def _check_greenboost_driver() -> Check:
    try:
        from greenboost_cli.greenboost.monitor import GreenBoostMonitor
        mon = GreenBoostMonitor()
        s   = mon.status
        if s and s.detected:
            t1 = round(s.fb_total_mb / 1024, 1)
            t2 = round(getattr(s, "ram_pool_mb", 0) / 1024, 1)
            t3 = round(getattr(s, "nvme_swap_total_mb", 0) / 1024, 1)
            parts = [f"T1 {t1} GB"]
            if t2:
                parts.append(f"T2 {t2} GB")
            if t3:
                parts.append(f"T3 {t3} GB")
            return Check(
                name="GreenBoost driver",
                ok=True,
                detail="  ·  ".join(parts),
            )
        return Check(
            name="GreenBoost driver",
            ok=False,
            detail="not detected (ioctl/sysfs/DKMS all failed)",
            fix="modprobe greenboost  (or reinstall kernel module)",
        )
    except Exception as e:
        return Check(
            name="GreenBoost driver",
            ok=False,
            detail=f"monitor error: {e}",
        )


def _check_diffusion() -> Check:
    try:
        import diffusers
        try:
            import bitsandbytes as bnb  # noqa: F401
            bnb_str = ""
        except ImportError:
            bnb_str = " · bitsandbytes missing (NF4 unavailable)"
        return Check(
            name="diffusion (FLUX/SD)",
            ok=True,
            detail=f"diffusers {diffusers.__version__}{bnb_str}",
        )
    except ImportError:
        return Check(
            name="diffusion (FLUX/SD)",
            ok=False,
            detail="not installed — /design-gen unavailable",
            fix="pip install greenboost-cli[diffusion]",
        )


def _check_markitdown() -> Check:
    try:
        from greenboost_cli.converters.markitdown_adapter import SUPPORTED_EXTENSIONS
        return Check(
            name="markitdown (/convert)",
            ok=True,
            detail=f"{len(SUPPORTED_EXTENSIONS)} supported formats",
        )
    except ImportError:
        return Check(
            name="markitdown (/convert)",
            ok=False,
            detail="not installed — /convert unavailable",
            fix="pip install greenboost-cli[convert]",
        )


def _check_design_skill_dir(settings: dict) -> Check:
    skill_dir = (
        settings.get("design_skill_dir")
        or os.environ.get("GB_DESIGN_SKILL_DIR", "")
        or str(Path.home() / "Dev/claude_workflow_sources/ui_design/ui-ux-pro-max-skill")
    )
    p = Path(skill_dir)
    if p.exists():
        n_csv = len(list(p.glob("*.csv")))
        if n_csv:
            short = str(p).replace(str(Path.home()), "~")
            return Check(
                name="design skill dir",
                ok=True,
                detail=f"{n_csv} CSV files at {short}",
            )
    return Check(
        name="design skill dir",
        ok=False,
        detail=f"not found: {skill_dir}",
        fix="set GB_DESIGN_SKILL_DIR or settings['design_skill_dir']",
    )


def _check_gb_synapse(settings: dict) -> Check:
    try:
        from greenboost_cli.slash_commands.backend_cmds import _import_gb_synapse
        gb_synapse = _import_gb_synapse()
    except ImportError as e:
        return Check(
            name="gb-synapse", ok=False, detail=f"cannot import: {e}",
            fix=gb_root_hint(),
        )
    try:
        d = gb_synapse.doctor(probe_feeders=False)
    except Exception as e:
        return Check(name="gb-synapse", ok=False, detail=f"doctor() failed: {e}")
    if not d["engine_installed"]:
        return Check(
            name="gb-synapse", ok=False, detail="engine not built",
            fix="sudo greenboost synapse build-engine",
        )
    if not settings.get("model"):
        return Check(
            name="gb-synapse", ok=False, soft=True, detail="no model configured",
            fix="/model <name>  or  /setup",
        )
    from greenboost_cli.slash_commands.backend_cmds import llamacpp_server_status
    st = llamacpp_server_status(settings)
    ok = st == "running"
    return Check(
        name="gb-synapse", ok=ok, detail=f"engine {d['engine_version']}  ·  server: {st}",
        fix="/llamaserve" if not ok else "",
    )


def _check_semble() -> Check:
    if shutil.which("semble"):
        return Check(name="semble (code search)", ok=True, detail="on PATH")
    try:
        r = subprocess.run(
            ["uvx", "--from", "semble[mcp]", "semble", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return Check(
                name="semble (code search)",
                ok=True,
                detail=f"available via uvx ({r.stdout.strip()})",
            )
    except Exception:
        pass
    return Check(
        name="semble (code search)",
        ok=False,
        detail="not found — MCP semantic code search unavailable",
        fix="pip install semble[mcp]",
    )


def _check_gb_quant() -> Check:
    gb_src = gb_py_root()
    gb_quant = gb_src / "gb_quant.py"
    if gb_quant.exists():
        return Check(
            name="gb-quant",
            ok=True,
            detail=f"found at {gb_src}",
        )
    return Check(
        name="gb-quant",
        ok=False,
        detail=f"not found at {gb_src}",
        fix=gb_root_hint(),
    )


# ── Check groups ───────────────────────────────────────────────────────────────

_GROUPS: list[tuple[str, list[Callable]]] = [
    ("Required", [
        _check_python,
    ]),
    ("Acceleration", [
        _check_torch_cuda,
        _check_greenboost_driver,
    ]),
    ("Intelligence", [
        _check_sentence_transformers,
        _check_rag_index,
        _check_markitdown,
    ]),
    ("Services", [
        _check_ollama,
        _check_mcp_module,
        # vllm and semble are settings-dependent, handled below
    ]),
    ("Extras", [
        _check_diffusion,
        _check_openai_sdk,
        _check_gb_quant,
        # design_skill_dir is settings-dependent, handled below
    ]),
]


# ── Display ────────────────────────────────────────────────────────────────────

def _render_check(c: Check) -> None:
    if c.ok:
        icon  = f"[{LIME}]✓[/]"
        color = GRAY
    elif c.soft:
        icon  = f"[{AMBER}]⚠[/]"
        color = AMBER
    else:
        icon  = f"[{RED}]✗[/]"
        color = RED

    console.print(
        f"  {icon}  [{color}]{c.name:<30}[/]  [{DIM}]{c.detail}[/]"
    )
    if not c.ok and c.fix:
        console.print(f"       [{DIM}]→ {c.fix}[/]")


# ── Command ────────────────────────────────────────────────────────────────────

def _doctor(args: str, session, settings: dict) -> None:
    fix_only = "--fix" in args

    # Build settings-dependent checks
    gb_synapse_check  = lambda: _check_gb_synapse(settings)
    design_dir_check  = lambda: _check_design_skill_dir(settings)

    groups_with_extras: list[tuple[str, list[Callable]]] = []
    for name, checks in _GROUPS:
        row = list(checks)
        if name == "Services":
            row += [gb_synapse_check, _check_semble]
        if name == "Extras":
            row += [design_dir_check]
        groups_with_extras.append((name, row))

    all_checks: list[Check] = []

    console.print()
    console.print(
        f"  [{TEAL}]◈  GreenBoost Doctor[/]"
        f"  [{DIM}]{'─' * 48}[/]"
    )

    for group_name, fns in groups_with_extras:
        group_checks = [fn() for fn in fns]
        all_checks.extend(group_checks)

        if fix_only:
            continue

        console.print()
        console.print(f"  [{VIOLET}]{group_name}[/]")
        for c in group_checks:
            _render_check(c)

    if fix_only:
        failing = [c for c in all_checks if not c.ok and c.fix]
        if not failing:
            console.print(f"\n  [{LIME}]All checks passed — nothing to fix.[/]")
        else:
            console.print(f"\n  [{AMBER}]Fix commands:[/]")
            for c in failing:
                console.print(f"  [{DIM}]{c.name}:[/]  [{GRAY}]{c.fix}[/]")
        console.print()
        return

    # Summary line
    n_pass  = sum(1 for c in all_checks if c.ok)
    n_warn  = sum(1 for c in all_checks if not c.ok and c.soft)
    n_err   = sum(1 for c in all_checks if not c.ok and not c.soft)
    w       = 60

    color = LIME if n_err == 0 and n_warn == 0 else (AMBER if n_err == 0 else RED)
    summary = f"  [{color}]{n_pass} passed[/]"
    if n_warn:
        summary += f"  [{AMBER}]·  {n_warn} warnings[/]"
    if n_err:
        summary += f"  [{RED}]·  {n_err} errors[/]"

    console.print()
    console.print(f"  [{DIM}]{'─' * w}[/]")
    console.print(summary)
    if n_err or n_warn:
        console.print(f"  [{DIM}]/doctor --fix   to print fix commands[/]")
    console.print()


register_command("doctor", _doctor, "Health diagnostics  (/doctor [--fix])")
