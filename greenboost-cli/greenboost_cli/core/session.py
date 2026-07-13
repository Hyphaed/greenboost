"""Conversation session state."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversationSession:
    """Mutable state for an ongoing conversation.

    messages use the provider-independent neutral format:
        {"role": "user",      "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}
    """
    messages:            list  = field(default_factory=list)
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    turn_count:          int   = 0
    # Plan mode: when True, the workflow middleware appends a directive telling
    # the model to confine writes to `plan_file`. See greenboost_cli/planning.
    plan_mode:           bool         = False
    plan_file:           Path | None  = None
    # Session display name (set via /name, shown in prompt and /status)
    name:                str | None   = None
    # Unix timestamp of session start (set in run_interactive, reset by /clear)
    _start_time:         float | None = field(default=None, repr=False)
    # Note queued by /note to be prepended to the next user message
    _pending_note:       str          = field(default="", repr=False)
    # Image attachments queued by /image for the next user message
    pending_attachments: list         = field(default_factory=list, repr=False)
    # PromptQueue wired in by run_interactive(); None on test/CLI sessions
    prompt_queue:        object | None = field(default=None, repr=False)
    # MCPRegistry attached when .mcp.json is found at startup
    mcp_registry:        object | None = field(default=None, repr=False)
