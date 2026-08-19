# gb_reclaim.py — selective GPU/T2/T3 reclaim

`greenboost clear memory-pool` used to be a blunt nuke: kill every process
holding `/dev/greenboost` open or using ≥512MB of GPU compute, filtered only
by a protected-comm allowlist (desktop shell, display servers, VM
GPU-passthrough processes). It had no notion of "the one stuck job I
actually want to kill" vs. "another genuinely-in-progress GreenBoost job on
this same node" — CLAUDE.md's own operating rule already called this out:
*"Never run it while other genuinely-in-progress GreenBoost work is on the
SAME node unless the owner explicitly authorizes it."*

`gb_reclaim.py` (repo root) is the shared classification + escalation logic
that fixes this, used by every surface that needs to reclaim GreenBoost
memory: the bash `greenboost clear memory-pool` command, the
`greenboost-orchestrator` MCP tools, and `greenboost-cli`'s own exit-time
cleanup.

## Classification

Every reclaim-candidate process (holds `/dev/greenboost` open, or uses
`>= kill_min_mb` of GPU compute per `nvidia-smi`) is put into one of three
buckets:

| Bucket | Meaning |
|---|---|
| `live` | `gb_synapse.ps()` is currently tracking this PID as a real, in-progress server. Never a reclaim target short of `scope="all"`. |
| `ambiguous` | Not tracked, but emitted a dataflux event within the last 10 minutes (`GB_RECLAIM_RECENCY_MINUTES` isn't a thing yet — see Future work) — probably legitimate (a torch/diffusion job, or a synapse process whose run-state file was lost), treat with more caution than plain residue. |
| `residue` | Not tracked, no recent activity — the orphaned-process case this module exists to target safely. |

Protected comm names (desktop shell, display servers, audio/session
daemons, VM GPU-passthrough processes) are filtered out **before**
classification and are never a target at any scope — `_gb_proc_is_protected()`
mirrors `greenboost_setup.sh`'s own `_gb_proc_is_protected` bash function
exactly (kept in sync by hand; there's no shared source between a shell
script and a Python module short of shelling out per check).

## Scope

`plan_reclaim(scope=...)` / `run_reclaim(scope=...)` take one of:

- **`"residue"` (default)** — orphaned processes only. Safe to run without
  checking for other in-progress work first; never touches a tracked
  gb-synapse server.
- **`"ambiguous"`** — residue + ambiguous.
- **`"all"`** — every non-protected candidate, including `live` — reproduces
  the old bash nuke's full blast radius. Never the default anywhere; always
  an explicit opt-in (`--all` on the CLI, `scope="all"` everywhere else).

## Escalation

1. Graceful unload API first, when one exists — Ollama's `/api/generate`
   with `keep_alive=0` (`_ollama_unload`, moved here from
   `greenboost-cli/greenboost_cli/rag/memory_pool.py`).
2. `SIGTERM` to every target, wait `term_wait_s` (default 2s) for a clean
   CUDA teardown.
3. `SIGKILL` any stragglers still alive.

`run_reclaim()` always emits a `kind="reclaim"` dataflux event with the
scope, targets, and outcome — visible via `dataflux_events`/
`dataflux_summary`/`dataflux_critic` regardless of which surface triggered
it.

## Surfaces

| Surface | Default | How to override |
|---|---|---|
| `greenboost clear memory-pool` (bash, `greenboost_setup.sh`) | `scope="residue"` | `greenboost clear memory-pool --all` |
| `greenboost-orchestrator` MCP: `reclaim_plan` / `reclaim_run` | `scope="residue"` | pass `scope="all"` |
| `greenboost-cli` REPL exit hook (`terminal/repl.py`) | `scope="residue"`, always | not overridable — exit-time cleanup only ever touches orphaned residue |

`reclaim_run` (MCP) is double-gated like every other actuating verb in
`gb_actuation.py`: dry-run (plan only, nothing killed) unless
`confirm=True` **and** the server env has `GB_ORCH_ACTUATE=1`. `reclaim_plan`
is a pure query, no gate.

## The old `gb clear-memory-pool` command

`greenboost-cli`'s own `gb clear-memory-pool` command (a narrower,
Ollama-only slice of this same idea, living in
`greenboost_cli/rag/memory_pool.py`) is retired — it duplicated a subset of
what the real command now covers via `gb_reclaim.py`. Running it prints a
one-release deprecation notice pointing at `sudo greenboost clear
memory-pool` instead of doing anything; the stub will be removed entirely
in the release after next.

## Future work

- The `ambiguous` bucket's recency window (10 minutes) is a constant in
  `gb_reclaim.py` (`DEFAULT_RECENCY_MINUTES`), not yet env-overridable.
- gb-synapse's own launched processes (llama-server, the proxy) never emit
  dataflux events about themselves, so an orphaned gb-synapse job
  classifies straight to `residue` rather than passing through `ambiguous`
  first — this is correct today (residue is exactly what should be
  reclaimed by default), but means the `ambiguous` bucket is currently only
  reachable by processes that self-report to dataflux (e.g. a torch/
  diffusion worker), not by anything gb-synapse launches.
