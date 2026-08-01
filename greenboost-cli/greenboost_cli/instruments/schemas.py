"""Tool JSON schemas sent to the AI API."""

INSTRUMENT_DEFINITIONS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": (
            "Write content to a file, creating parent directories as needed. "
            "Overwrites the file completely — use Edit instead to change specific lines. "
            "Prefer Edit for existing files; use Write only for new files or full rewrites."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "content":   {"type": "string", "description": "Full file content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": (
            "Execute a shell command. Returns stdout+stderr. "
            "`cd` commands persist across calls within the same session. "
            "Default timeout is 120 s; pass timeout= for long operations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 120)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep). Use for exact literal strings, regex patterns, or enumerating all occurrences. For semantic 'where is X defined' / 'how is X implemented' queries, prefer Semble.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":          {"type": "string", "description": "Regex pattern"},
                "path":             {"type": "string", "description": "File or directory to search"},
                "glob":             {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":      {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":          {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": (
            "Search the web via DuckDuckGo and return top results (title + URL + snippet). "
            "Use for current events, package docs, or any information that may be more "
            "recent than your training data. Follow up with WebFetch to read full pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "Screenshot",
        "description": (
            "Load a URL in headless Chromium and save a PNG screenshot to disk "
            "(e.g. a running dev server, http://localhost:1420). You have no "
            "vision capability, so you cannot judge the resulting image "
            "yourself — use this to CAPTURE a UI state for a vision-capable "
            "reviewer (a human, or another AI with vision), or to confirm a "
            "page loads without error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url":         {"type": "string"},
                "output_path": {"type": "string", "description": "Absolute path to write the PNG"},
                "width":       {"type": "integer", "description": "Viewport width (default 1280)"},
                "height":      {"type": "integer", "description": "Viewport height (default 800)"},
                "full_page":   {"type": "boolean", "description": "Capture the full scrollable page, not just the viewport"},
            },
            "required": ["url", "output_path"],
        },
    },
    {
        "name": "MemoryRead",
        "description": (
            "Read from persistent CLAUDE.md memory files (project and/or global). "
            "Use to recall project conventions, decisions, and notes saved in previous sessions. "
            "Always check memory before starting on a new task if it might have relevant context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Optional: absolute path to a specific CLAUDE.md. Default reads both project and global."},
            },
            "required": [],
        },
    },
    {
        "name": "MemoryWrite",
        "description": (
            "Write a persistent note to CLAUDE.md memory. Use to save important project conventions, "
            "architectural decisions, test commands, and context that should persist across sessions. "
            "Call this when you discover something the user would want you to remember."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key":     {"type": "string", "description": "Section heading (e.g. 'Test Commands', 'Key Conventions', 'Architecture')"},
                "content": {"type": "string", "description": "Content to write under this section"},
                "scope":   {"type": "string", "enum": ["project", "global"],
                            "description": "project = ./CLAUDE.md, global = ~/.claude/CLAUDE.md"},
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "TodoRead",
        "description": (
            "Read the current task list for this session. Returns a JSON array of todo items. "
            "Call this to check what's pending before starting work or to update task status. "
            "Use TodoWrite to update the list after reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "TodoWrite",
        "description": (
            "Write the task list for this session. Replaces the entire list atomically. "
            "Use this to track tasks: set status to 'in_progress' when starting, "
            "'completed' when done, 'pending' for not yet started. "
            "Always have at most one 'in_progress' task at a time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Complete list of todos (replaces current list)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":       {"type": "string", "description": "Unique short identifier"},
                            "content":  {"type": "string", "description": "Task description"},
                            "status":   {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["id", "content", "status", "priority"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "Semble",
        "description": (
            "Semantic code search (local, fast, offline). Prefer this for 'where is X defined', "
            "'how is X implemented', or locating code by intent — returns file:line snippets "
            "using ~98% fewer tokens than grep+read. Use Grep instead for exact literal "
            "strings/regex or when you need to enumerate every occurrence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "Natural-language or code query"},
                "repo":    {"type": "string", "description": "Repo path to search (default: cwd)"},
                "top_k":   {"type": "integer", "description": "Number of results (default: 5)"},
                "content": {
                    "type": "string",
                    "enum": ["code", "docs", "config", "all"],
                    "description": "Content type to search (default: code)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "AskUserQuestion",
        "description": (
            "Ask the user one or more clarifying questions through an interactive wizard. "
            "Use this INSTEAD of writing questions as plain text — the user will see a "
            "stepped picker ([1/N], [2/N], …) with labeled options and may also type a "
            "free-form answer. Call this whenever you need a decision or missing detail "
            "before proceeding. The user's answers are returned so you can continue "
            "automatically in the same turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "List of questions to show as steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question text shown to the user.",
                            },
                            "header": {
                                "type": "string",
                                "description": "Short chip label shown in the step counter (≤12 chars).",
                            },
                            "multiSelect": {
                                "type": "boolean",
                                "description": "If true, the user may pick several options.",
                            },
                            "options": {
                                "type": "array",
                                "description": (
                                    "2–4 choices. An 'Other / free text' option is always "
                                    "appended automatically."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label":       {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label"],
                                },
                            },
                        },
                        "required": ["question", "header", "options"],
                    },
                },
            },
            "required": ["questions"],
        },
    },
    {
        "name": "ToolSearch",
        "description": (
            "Fetch the full parameter schema for one or more MCP tools by name or "
            "keyword. When many MCP servers are connected, most MCP tools are "
            "advertised to you with only a name and short description (no "
            "input_schema) to keep the prompt small — call ToolSearch first to get "
            "the real argument names/types before calling an unfamiliar mcp__ tool. "
            "Builtin tools (Read, Write, Bash, Edit, Grep, Glob, …) already have "
            "full schemas and never need this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Tool name (exact or partial) or a keyword from its "
                        "description, e.g. 'generate_image' or 'mcp__forge3d'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]
