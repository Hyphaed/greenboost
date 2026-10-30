"""Filesystem containment for Bash, so `workspace_roots` actually contains.

`ToolPolicy.workspace_roots` used to jail only Write and Edit. A worker with
Bash allowed could `rm -rf ~` or write anywhere, so the field read as containment
and wasn't. When a policy carries workspace roots, Bash now runs inside
bubblewrap: the whole filesystem is visible read-only, the workspace roots are
writable, /tmp is private, and the process dies with its parent.

What this does NOT do: it doesn't restrict reads (the CLI has to read the repo
and system state), and it doesn't touch the network. Network policy is a
separate layer.

Fail closed. If a policy asks for containment and bubblewrap isn't installed,
Bash is refused with the install command. Running the command unjailed and
saying nothing is exactly the false assurance this module exists to remove.
"""
from __future__ import annotations

import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

_tls = threading.local()


class SandboxUnavailable(RuntimeError):
    """A policy requires containment and this host can't provide it."""


@contextmanager
def sandbox(roots):
    """Mark the calling thread's shell calls as jailed to `roots` (empty = off)."""
    prev = getattr(_tls, "roots", ())
    _tls.roots = tuple(str(r) for r in (roots or ()))
    try:
        yield
    finally:
        _tls.roots = prev


def active_roots() -> tuple:
    return getattr(_tls, "roots", ())


def bwrap_argv(command: str, roots, cwd: str) -> list[str]:
    """argv that runs `command` under bash with only `roots` writable."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable(
            "this agent's policy confines Bash to its workspace, and bubblewrap "
            "(bwrap) isn't installed, so the command was not run. "
            "Install it with: sudo apt install bubblewrap")
    argv = [bwrap, "--die-with-parent", "--unshare-pid",
            "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp"]
    for r in roots:
        argv += ["--bind", str(r), str(r)]
    # A cwd outside every root is still readable; Bash just can't write there.
    chdir = cwd if cwd and Path(cwd).is_dir() else str(roots[0])
    return argv + ["--chdir", chdir, "/bin/bash", "-c", command]
