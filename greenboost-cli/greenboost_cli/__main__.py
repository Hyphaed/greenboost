"""
GreenBoost CLI — entry point.

Usage:
  greenboost-cli [options] [prompt]
  python -m greenboost_cli [options] [prompt]

Commands:
  web                  Open the web dashboard at http://localhost:7821
  setup                Run the interactive setup wizard
  help                 Show this help
  version              Print version and exit

Options:
  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model for this session (gb-synapse manifest name)
  --accept-all         Never ask permission (dangerous)
  --verbose            Show reasoning tokens and tool details
  --setup              Run the interactive setup wizard
  --version            Print version and exit
  -h, --help           Show this help

Examples:
  greenboost-cli web                    # open web dashboard
  gb "fix the bug in main.py"           # run a one-shot prompt
  gb -m qwen3-coder "list files"        # use a specific gb-synapse model
  gb                                    # start interactive REPL

Every model runs through gb-synapse — GreenBoost's HuggingFace-pull +
Ollama-index + cluster-distributed llama.cpp serving layer. No cloud
credentials, no other backend.
"""
from __future__ import annotations

import sys
import argparse

VERSION = "1.0.0"

# Apply GreenBoost + PyTorch env before any torch.cuda initialisation.
# Must happen at module import time so PYTORCH_CUDA_ALLOC_CONF is set before
# any other module triggers CUDA lazy init.
try:
    from greenboost_cli.greenboost.gb_torch import apply_gb_torch_env
    apply_gb_torch_env()
except ImportError:
    pass  # greenboost module not installed — non-fatal
except Exception as _e:
    import sys as _sys
    print(f"Warning: GreenBoost env setup failed: {_e}", file=_sys.stderr)


def main() -> None:
    # Headless subcommand fast-path: `gb rag-search`, `gb compress`, etc.
    # These bypass the REPL so they can be called from scripts (e.g.
    # optimal-claude's gb_bridge). They run before the main argparse because
    # they accept their own flags.
    if len(sys.argv) >= 2:
        from greenboost_cli.cli_headless import HEADLESS_SUBCOMMANDS, dispatch
        name = sys.argv[1]
        # Two-word form: `gb rag update --all` → `gb rag-update --all`
        if name == "rag" and len(sys.argv) >= 3:
            candidate = f"rag-{sys.argv[2]}"
            if candidate in HEADLESS_SUBCOMMANDS:
                sys.exit(dispatch(candidate, sys.argv[3:]))
        # Two-word form: `gb crag add ./docs` → `gb crag-add ./docs`
        if name == "crag" and len(sys.argv) >= 3:
            candidate = f"crag-{sys.argv[2]}"
            if candidate in HEADLESS_SUBCOMMANDS:
                sys.exit(dispatch(candidate, sys.argv[3:]))
        if name in HEADLESS_SUBCOMMANDS:
            sys.exit(dispatch(name, sys.argv[2:]))

    parser = argparse.ArgumentParser(
        prog="greenboost-cli",
        description="GreenBoost CLI — unified AI coding assistant for the terminal",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="Initial prompt (non-interactive)")
    parser.add_argument(
        "-p", "--print", "--print-output",
        dest="print_mode", action="store_true",
        help="Non-interactive mode: run prompt and exit",
    )
    parser.add_argument("-m", "--model", help="Override model (gb-synapse manifest name)")
    parser.add_argument(
        "--accept-all", action="store_true",
        help="Never ask permission (accept all operations)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show reasoning + token counts")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")

    args = parser.parse_args()

    if args.version:
        print(f"greenboost-cli v{VERSION}")
        sys.exit(0)

    if args.help:
        print(__doc__)
        sys.exit(0)

    from greenboost_cli.environment.settings import load_settings, save_settings, SETTINGS_PATH
    from greenboost_cli.inference.registry import BACKEND_REGISTRY

    settings = load_settings()

    # First launch or explicit --setup
    if args.setup or not SETTINGS_PATH.exists():
        from greenboost_cli.wizard.setup import run_wizard
        run_wizard(settings)
        save_settings(settings)
        if args.setup:
            return

    # Apply CLI overrides
    if args.model:
        _m = args.model
        # Rewrite "gb-synapse:model" → "gb-synapse/model" — the CLI accepts
        # either separator, only the slash form is stored.
        if "/" not in _m and ":" in _m:
            _prefix, _, _rest = _m.partition(":")
            if _prefix in BACKEND_REGISTRY:
                _m = f"{_prefix}/{_rest}"
        settings["model"] = _m
    if args.accept_all:
        settings["permission_mode"] = "accept-all"
    if args.verbose:
        settings["verbose"] = True

    initial = " ".join(args.prompt) if args.prompt else None

    # Intercept bare meta-subcommands passed as positional args
    # e.g. "gb help", "gb version", "gb setup" — don't send to AI
    if initial:
        _meta = initial.strip().lower()
        if _meta in ("help", "--help", "-h"):
            print(__doc__)
            sys.exit(0)
        if _meta in ("version", "--version", "-v"):
            print(f"greenboost-cli v{VERSION}")
            sys.exit(0)
        if _meta in ("setup", "--setup"):
            from greenboost_cli.wizard.setup import run_wizard
            run_wizard(settings)
            save_settings(settings)
            sys.exit(0)
        if _meta in ("web", "dashboard"):
            try:
                from greenboost_cli.dashboard.server import start_server
                start_server()
            except Exception as e:
                from greenboost_cli.terminal.theme import emit_err
                emit_err(f"Dashboard error: {e}")
            sys.exit(0)
        if _meta in ("status", "gb-status"):
            try:
                from greenboost_cli.slash_commands.greenboost_cmds import cmd_gb_status
                cmd_gb_status("", None, settings)
            except Exception as e:
                from greenboost_cli.terminal.theme import emit_err
                emit_err(f"Status error: {e}")
            sys.exit(0)
        if _meta in ("vitals", "gb-vitals"):
            try:
                from greenboost_cli.slash_commands.greenboost_cmds import cmd_gb_vitals
                cmd_gb_vitals("", None, settings)
            except Exception as e:
                from greenboost_cli.terminal.theme import emit_err
                emit_err(f"Vitals error: {e}")
            sys.exit(0)
        if _meta in ("serve", "llamaserve"):
            try:
                from greenboost_cli.slash_commands.backend_cmds import cmd_llamaserve
                rest = " ".join(args.prompt[1:]) if args.prompt and len(args.prompt) > 1 else ""
                cmd_llamaserve(rest, None, settings)
            except Exception as e:
                from greenboost_cli.terminal.theme import emit_err
                emit_err(f"llamaserve error: {e}")
            sys.exit(0)

    if args.print_mode and not initial:
        from greenboost_cli.terminal.theme import emit_err
        emit_err("--print requires a prompt argument")
        sys.exit(1)

    # Auto-start gb-synapse if the configured model isn't already being served.
    _model = settings.get("model", "")
    if _model:
        try:
            from greenboost_cli.slash_commands.backend_cmds import _llamacpp_running_pid, cmd_llamaserve
            if not _llamacpp_running_pid(settings):
                from greenboost_cli.terminal.theme import emit_info, emit_warn, VIOLET
                _mid = _model.split("/", 1)[1] if "/" in _model else _model
                emit_info(f"Starting gb-synapse for [{VIOLET}]{_mid}[/] …")
                cmd_llamaserve("start", None, settings)
        except Exception as _e:
            from greenboost_cli.terminal.theme import emit_warn
            emit_warn(f"gb-synapse auto-start error: {_e}")

    from greenboost_cli.terminal.repl import run_interactive
    run_interactive(settings, initial_prompt=initial)


if __name__ == "__main__":
    main()
