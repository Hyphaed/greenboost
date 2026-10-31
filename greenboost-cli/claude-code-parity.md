# greenboost-cli vs Claude Code: parity audit (2026-10-29)

greenboost-cli only talks to a model served on this machine (gb-synapse, Qwen3.8-35B-A3B on one
12 GB RTX 5070). Features that exist for accounts, billing, remote sessions or IDE links are out of
scope on purpose. Everything else is tracked here.

## Already there before this audit (114 slash commands, 20 tools)

Plan mode and `/plan*`, todos (`TodoWrite`, `/todo`), subagents (`Delegate`, `/agent`, `/roles`),
MCP (`/mcp*`), skills (`/skill-*`, `/find-skills`), hooks (config in `~/.greenboost_cli/hooks.json`),
`/init`, `/git-review`, `/git-pr`, `/commit`, `/doctor`, `/compact`, `/context`, `/cost`, `/resume`,
`/fork`, `/undo`, `/revert`, `/permissions`, `/memories`, `/remember`, headless mode (`-p`),
background shells (`instruments/background.py`, `TaskOutput`, `TaskStop`), image paste, `ctrl+o`,
`ctrl+p` pause, `ctrl+n` non-stop, `ctrl+y` auto-answer, queued prompts, WebFetch/WebSearch,
`Semble` code search, ai-forge (`/forge`), the factory, design commands.

## Added in this pass

| Feature | Notes |
|---|---|
| `!cmd` inline shell | output sanitised, enters the context |
| `@path` file / dir mentions | Tab completion, binary skipped, 20k char cap |
| Markdown slash commands | `.greenboost/commands`, `~/.greenboost_cli/commands`; `$ARGUMENTS`; `/commands` lists them |
| Built-in prompt commands | `/security-review`, `/review`, `/explain` |
| `# note` | one-line memory through `/remember` |
| `ctrl+g` | edit the prompt in `$EDITOR` |
| `/export`, `/copy` | conversation to markdown, last reply to the clipboard |
| `/hooks`, `/vim`, `/think` | list hooks, vi editing mode, server-side thinking toggle |
| `/rewind /memory /usage /todos /rename /tasks /stats` | Claude Code names for existing commands |
| `/gpu` + load preflight | card usage, who holds it (ai-forge, FLUX, studio, Ollama), warns before `/llamaserve load`; never kills anything |
| Live activity line, pinned bottom bar, model state | see git history of `terminal/` |
| Friendly "model not loaded" error | points at `/llamaserve load`, `/retry` |

| Per-pattern permission rules | `/permissions allow\|deny\|remove\|rules`, `Tool(pattern)`; deny wins in every mode; chained shell commands can't ride an allow rule |
| `MultiEdit`, `NotebookEdit` | all-or-nothing multi edit; notebook cell replace / insert / delete |
| Double-Esc rewind picker | `/rewind`; `/rewind last` is the old undo |
| Custom status line | `/statusline <command>` (gets `{model, cwd}` on stdin) |
| GPU owner in the bottom bar | `GPU: FLUX generation 8.1G` when another heavy job holds the card |

| `/studio` | one-shot ask of the SVG Animation Studio's MCP server (`gpu_status`, `list_projects`); no port needed |
| `/output-style` | `concise`, `explanatory`, `learning`, or your own in `.greenboost/output-styles/*.md`; appended to the system prompt |
| `/pr-comments` | built-in prompt command, uses `gh` |

| Markdown subagents | `/agents`, `/agent <name> <task>`, `Delegate(agent=)` (`subagent_type=` accepted as an alias; an unknown name runs a general delegate); `.greenboost/agents`, `~/.greenboost_cli/agents`, then Claude Code's `.claude/agents`, `$CLAUDE_CONFIG_DIR/agents`, `~/.claude/agents`; tools allowlist honoured; one local model, a Claude `model:` line is ignored |

## Added 2026-10-30: it reads the same configuration Claude Code does

| Feature | Notes |
|---|---|
| MCP servers | merged from `~/.claude.json`, `$CLAUDE_CONFIG_DIR/.claude.json` (+ `projects[cwd]`) and the nearest `.mcp.json`, later wins; `${VAR}`, `disabled`, own-server skip, greenboost mirror de-dup. 16 of 16 servers, 352 tools, connected in 3.5 s from a folder with no `.mcp.json` |
| MCP client | servers connect in parallel with a per-server timeout (`GB_MCP_INIT_TIMEOUT`), run in their own process group (closed whole on exit), answer `ping` and `roots/list`, no 200 KB reply cap; `/mcp-servers` shows state, why a server failed and what the prompt really carries; `/mcp-servers reconnect [name]`, `/mcp-servers log <name>`; `GB_MCP_AUTOCONNECT=0` skips it; `GB_MCP_EXCLUDE`/`GB_MCP_ONLY` choose which servers start; `/mcp-add`, `/mcp-import`, `/mcp-remove` add and remove servers live (`~/.greenboost_cli/mcp.json`); plan mode runs MCP tools that declare `readOnlyHint` (plus a small allowlist, `plan_mode_allow_mcp`); `ToolSearch` ranks by words and accepts `select:a,b` |
| Skills | project `.claude/skills`, active account, global, installed plugins (`plugin:skill`), in Claude Code's precedence; a bounded skill index in the system prompt, ranked by the project's stack; matched bodies go before the user turn so the prompt cache survives |
| `WebSearch`, `WebFetch` | search tries DuckDuckGo lite then html and reports a bot challenge as an error; fetch sends a descriptive User-Agent, names the bot wall in a 403, retries in a headless browser by default (`GB_WEBFETCH_BROWSER=0` disables) and honours `prompt` on long pages |
| `/exit` | clean terminal first, then MCP servers closed, the configured model unloaded (`/exit keep`, `GB_CLI_KEEP_MODEL=1` or `"unload_model_on_exit": false` keep it), pool released; Ctrl-D is the same path |
| Cancel (Esc, Ctrl-C) | stops the turn before a request is sent, and closes the request while the model reads the prompt (1.4 s measured; it never completed before) |
| `/huggingface-token` | saves the token to the standard hub file and gb-synapse's, owner-only; the hub's anonymous-download warning is replaced by one that says what to type |
| `/rag-reembed`, `GB_EMBED_MODEL`, `GB_EMBED_DEVICE` | Qwen3-Embedding-0.6B embedder: queries on the CPU (87 to 96 ms), bulk indexing on the GPU when it has room, resumable and atomic re-embed of the whole index; `scripts/rag_reembed.py` runs it unattended |

## Still missing

1. `/theme`, `/terminal-setup`. Compact output moved from ctrl+i (which is Tab, so completion never fired) to ctrl+q; ctrl+t shows the task board; ctrl+r history search verified.
2. MCP rules match the tool name and the first argument (`mcp__x__*(repo:acme/*)`); matching on a named argument is not built.
3. Large browser snapshot/network dumps are cut at 6,000 characters, not summarised by a sub-model; skill `allowed-tools` is not honoured; the legacy two-endpoint MCP SSE transport is tried as streamable HTTP.
4. The studio's `list_projects` tool fails on its side ("reading 'map'"), so `/studio` shows only the GPU summary.
