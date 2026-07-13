"""
Git workflow slash commands.

/commit          — stage all changes and create a conventional commit
/git-review      — review staged/unstaged diff before committing
/git-clean       — delete local branches that are gone on the remote
/git-pr          — create a pull request for the current branch
"""
from __future__ import annotations

import subprocess

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, LIME, AMBER,
    emit_ok, emit_warn, emit_err, emit_info,
)


def _git_context() -> str:
    """Return a compact git-state block for use in prompts."""
    def _run(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""

    status  = _run(["git", "status", "--short"])
    diff    = _run(["git", "diff", "HEAD"])[:4000]   # cap diff size
    branch  = _run(["git", "branch", "--show-current"])
    log     = _run(["git", "log", "--oneline", "-10"])
    return (
        f"Current branch: {branch}\n\n"
        f"Git status:\n{status or '(clean)'}\n\n"
        f"Recent commits:\n{log or '(none)'}\n\n"
        f"Diff (HEAD):\n{diff or '(no changes)'}"
    )


def cmd_commit(args: str, session, settings: dict) -> bool:
    """Stage all changes and create a conventional commit via the model."""
    from greenboost_cli.terminal.repl import process_query

    ctx = _git_context()
    prompt = (
        "Create a conventional git commit for the current changes.\n\n"
        f"{ctx}\n\n"
        "Instructions:\n"
        "1. Run `git add -A` to stage all changes\n"
        "2. Analyse the diff and write a concise commit message following Conventional Commits "
        "(feat/fix/refactor/chore/docs/test: <summary>)\n"
        "3. Run `git commit -m \"<message>\"` — single commit, no amend\n"
        "4. Report the commit hash when done\n"
        "Do not push or create a PR unless asked."
    )
    if args.strip():
        prompt += f"\n\nUser note: {args.strip()}"

    process_query(prompt, session, settings)
    return True


def cmd_git_review(args: str, session, settings: dict) -> bool:
    """Review current git diff and give improvement suggestions."""
    from greenboost_cli.terminal.repl import process_query

    ctx = _git_context()
    aspects = args.strip() or "correctness, code quality, missing tests, potential bugs"
    prompt = (
        f"Review the following git changes. Focus on: {aspects}.\n\n"
        f"{ctx}\n\n"
        "Provide a concise review covering:\n"
        "- What the changes do\n"
        "- Any bugs, edge cases, or correctness issues\n"
        "- Code quality and readability concerns\n"
        "- Suggestions (not mandatory rewrites)\n"
        "Keep it actionable. Use tools (Read, Grep) if you need more context."
    )
    process_query(prompt, session, settings)
    return True


def cmd_git_clean(args: str, session, settings: dict) -> bool:
    """Delete local branches that are gone on the remote."""
    from greenboost_cli.terminal.repl import process_query

    prompt = (
        "Clean up local git branches whose remote tracking branch has been deleted.\n\n"
        "Steps:\n"
        "1. Run `git fetch --prune` to sync remote state\n"
        "2. Run `git branch -v` and identify branches marked [gone]\n"
        "3. For each [gone] branch, check if a worktree is attached "
        "(`git worktree list`) — remove the worktree first if so\n"
        "4. Delete each [gone] branch with `git branch -D <name>`\n"
        "5. Report which branches were deleted and which were skipped (e.g. current branch)\n"
        "Do NOT delete the current branch or any branch without [gone] status."
    )
    process_query(prompt, session, settings)
    return True


def cmd_git_pr(args: str, session, settings: dict) -> bool:
    """Create a pull request for the current branch using gh CLI."""
    from greenboost_cli.terminal.repl import process_query

    ctx = _git_context()
    extra = f"\nUser instructions: {args.strip()}" if args.strip() else ""
    prompt = (
        "Create a pull request for the current branch using the `gh` CLI.\n\n"
        f"{ctx}{extra}\n\n"
        "Instructions:\n"
        "1. Check if gh is available: `gh --version`\n"
        "2. Make sure the branch is pushed: `git push -u origin HEAD` if needed\n"
        "3. Write a clear PR title (≤70 chars) and body summarising WHAT changed and WHY\n"
        "4. Run `gh pr create --title \"...\" --body \"...\"`\n"
        "5. Return the PR URL\n"
        "Do NOT push to main/master directly."
    )
    process_query(prompt, session, settings)
    return True


# ── Registration ──────────────────────────────────────────────────────────────

register_command("commit",      cmd_commit,     "Stage and commit all changes via the model")
register_command("git-review",  cmd_git_review, "Review current diff before committing")
register_command("git-clean",   cmd_git_clean,  "Delete branches marked [gone] on remote")
register_command("git-pr",      cmd_git_pr,     "Create a pull request for the current branch")
