"""Shell command safety classification.

Two approval tiers are exported:

  is_readonly_command(cmd)   — "auto" mode: only pure read-only ops, no interpreters.
  is_autonomous_safe(cmd)    — "autonomous" mode: adds test/build/lint/git-write ops.
                               The user has explicitly consented to unattended execution.

Both tiers always refuse chain operators and a hard list of destructive commands.
"""
from __future__ import annotations

# Commands that are always safe to run without permission prompts.
#
# SECURITY: interpreters (python, node, perl, ruby, …) are deliberately
# excluded from the *read-only* tier — a single `python -c "..."` call with
# no shell chaining would still execute arbitrary code.  They appear in
# AUTONOMOUS_EXTRA_PREFIXES for the autonomous tier instead.
READONLY_CMD_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "grep ", "rg ", "ag ", "fd ",
    "pip show", "pip list", "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
)

# Shell operators that allow command chaining or output redirection.
# Blocked in BOTH tiers — chaining can compose safe prefixes into dangerous sequences.
# Note: "&&" is NOT in this list because is_readonly_command() handles it specially:
# a chain of all-readonly commands joined by && is itself considered readonly.
CHAIN_OPERATORS = (";", "||", "|", "`", "$(", ">", "\n")

# find flags that can delete or overwrite files — always unsafe.
_DESTRUCTIVE_FIND_FLAGS = (
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprint0", "-fprintf",
)

# Commands that are ALWAYS refused, even in autonomous mode.
_HARD_BLOCKED_PREFIXES = (
    "git push",           # never publish without the user present
    "rm -rf", "rm -r ",   # mass deletion
    "sudo rm", "sudo dd",
    "rmdir ",
    "DROP ", "DROP TABLE", "DELETE FROM",   # SQL destructive ops
    "kubectl delete",
    "terraform destroy",
    "shred ", "wipe ",
)

# Extra commands auto-approved in autonomous-coding mode (user has consented).
AUTONOMOUS_EXTRA_PREFIXES = (
    # Interpreters — run scripts, not arbitrary one-liners (chain guard still applies)
    "python ", "python3 ", "python -m ",
    "node ", "bun ", "deno ",
    "ruby ", "perl ",
    # Test runners
    "pytest", "py.test",
    "npm test", "npm run ", "npx ",
    "cargo test", "cargo build", "cargo run ", "cargo check", "cargo clippy",
    "go test", "go build", "go run ", "go vet",
    "mix test", "mix ",
    "jest ", "vitest ", "mocha ",
    # Build / compile
    "make ", "cmake ", "ninja ",
    "./gradlew ", "gradlew ",
    "mvn ", "./mvnw ",
    "tsc ", "esbuild ", "vite ",
    # Package / dependency managers
    "pip install", "pip3 install", "pip uninstall",
    "npm install", "npm ci", "npm update",
    "yarn ", "pnpm ",
    "poetry install", "poetry run ", "poetry add",
    "uv pip", "uv run ", "uv sync",
    "cargo add", "cargo update",
    # Linters / formatters (safe — never delete files)
    "black ", "ruff ", "flake8 ", "mypy ", "pylint ",
    "eslint ", "prettier ", "tsc --",
    "rustfmt", "cargo fmt",
    "shellcheck ",
    # Non-destructive file ops
    "mkdir ", "cp ", "mv ", "touch ",
    # Git write ops (commits, staging — but NOT push, covered by hard-block list)
    "git add", "git commit", "git stash",
    "git checkout ", "git switch ",
    "git reset --soft", "git reset --mixed",
    "git fetch", "git pull",
    "git merge", "git rebase",
    "git tag ",
    # Docs generation
    "sphinx-build", "mkdocs ", "pdoc ",
)


def _is_safe_find(cmd: str) -> bool:
    """Return True if this find command only lists/prints — no destructive flags."""
    import shlex
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    return not any(t in _DESTRUCTIVE_FIND_FLAGS for t in tokens)


def _is_part_readonly(c: str) -> bool:
    """Check a single (non-chained) command fragment for read-only safety."""
    c = c.strip()
    if not c:
        return True
    # cd is always safe — it only affects the current subshell's cwd
    if c == "cd" or c.startswith("cd "):
        return True
    if c.startswith("find ") or c == "find":
        return _is_safe_find(c)
    return any(c.startswith(prefix) for prefix in READONLY_CMD_PREFIXES)


def _is_part_autonomous(c: str) -> bool:
    """Check a single (non-chained) command fragment for autonomous-mode safety."""
    c = c.strip()
    if not c:
        return True
    if c == "cd" or c.startswith("cd "):
        return True
    if any(c.startswith(p) for p in _HARD_BLOCKED_PREFIXES):
        return False
    if c.startswith("find ") or c == "find":
        return _is_safe_find(c)
    if any(c.startswith(p) for p in READONLY_CMD_PREFIXES):
        return True
    return any(c.startswith(p) for p in AUTONOMOUS_EXTRA_PREFIXES)


def is_readonly_command(cmd: str) -> bool:
    """Return True if cmd is safe to auto-approve in *auto* permission mode.

    Conservative: read-only operations only. Interpreters and build tools
    require explicit approval or autonomous mode.

    && chains are allowed when EVERY part is independently read-only (e.g.
    "cd /repo && git diff --stat HEAD"). All other chain operators are blocked.
    """
    c = cmd.strip()
    if any(op in c for op in CHAIN_OPERATORS):
        return False
    if "&&" in c:
        return all(_is_part_readonly(part) for part in c.split("&&") if part.strip())
    return _is_part_readonly(c)


def is_autonomous_safe(cmd: str) -> bool:
    """Return True if cmd should auto-approve in *autonomous* permission mode.

    Superset of is_readonly_command(): adds interpreters, test runners, build
    tools, package managers, linters, and git write operations (excluding push).

    && chains are allowed when every part passes autonomous-mode checks.
    Hard-blocked commands and other chain operators still apply.
    """
    c = cmd.strip()
    if any(op in c for op in CHAIN_OPERATORS):
        return False
    if any(c.startswith(p) for p in _HARD_BLOCKED_PREFIXES):
        return False
    if "&&" in c:
        return all(_is_part_autonomous(part) for part in c.split("&&") if part.strip())
    return _is_part_autonomous(c)
