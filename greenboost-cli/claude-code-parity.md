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

| Markdown subagents | `/agents`, `/agent <name> <task>`, `Delegate(agent=)`; `.greenboost/agents` and `~/.greenboost_cli/agents`; tools allowlist honoured; one local model, no per-agent model |

## Still missing

1. `/theme`, `/terminal-setup`. Compact output moved from ctrl+i (which is Tab, so completion never fired) to ctrl+q; ctrl+t shows the task board; ctrl+r history search verified.
2. MCP rules match the tool name and the first argument (`mcp__x__*(repo:acme/*)`); matching on a named argument is not built.
3. The studio's `list_projects` tool fails on its side ("reading 'map'"), so `/studio` shows only the GPU summary.
