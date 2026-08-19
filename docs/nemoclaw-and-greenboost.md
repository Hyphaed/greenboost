# NemoClaw and GreenBoost — not a runtime, a client

NVIDIA NemoClaw (`https://github.com/NVIDIA/NemoClaw`, alpha, Apache-2.0) was
audited as a possible dependency for greenboost-cli and GreenBoost core. This
doc carries the verdict so a future session doesn't re-litigate it, plus the
one integration that's actually worth doing: pointing NemoClaw's managed
inference at gb-synapse as an OpenAI-compatible provider.

## The verdict

| | GreenBoost | NemoClaw |
|---|---|---|
| What it is | CUDA memory/compute orchestrator — extends usable GPU memory across VRAM, DDR, NVMe, and remote GPUs | Sandbox + lifecycle runtime for coding agents (OpenShell containers, network egress policy, credential handling, managed inference) |
| Core strength | CUDA allocation, unified memory, PCIe paging, VRAM oversubscription | Sandboxing, runtime policy enforcement, agent orchestration, secrets management |
| Overlaps with greenboost-cli | — | ~90% of its own agent layer: turn loop, tool registry, approval tiers, hooks, skills, MCP client/server, sub-agent delegation |

**Not a runtime, not a dependency.** greenboost-cli exists to actuate host
state — `/dev/greenboost`, `/run/greenboost/shim_stats`, the LD_PRELOAD shim,
the MCP actuation verbs, SSH to feeders. Putting it inside an OpenShell
sandbox severs it from exactly the state it manages: sandbox GPU access is
passthrough-only and fixed at creation, so a sandboxed greenboost-cli could
observe a GPU but couldn't actuate the tiering/cluster levers that are its
whole point.

**The premise that prompted the audit didn't survive contact with this repo
either.** A prior outside analysis recommended building "a stable CLI/API, a
telemetry endpoint, a plugin architecture" before considering any NemoClaw
integration — all three already exist: 4 MCP servers with 60+ tools, all
actuation normalized through one `gb_actuation.VERBS` dict; Prometheus `:9742`
+ JSON `:8799/api/*.json` telemetry; `optimize_inference`'s gated closed loop
with auto-rollback. See `CLAUDE.md`'s Central MCP row and the A2A gateway
(`docs/a2a-interop.md`) for the actuation surface a NemoClaw-style agent would
otherwise have wanted to bolt on.

**What IS worth taking**, as design references only, zero dependency:
NemoClaw's per-agent tool-scoping model and its bearer-token proxy pattern.
The latter is what shaped `gb_synapse_api.py`'s auth middleware (see
`workflow/architecture.md`'s `gb_synapse_api.py` auth section).

## The value that does exist: GreenBoost as a provider

NemoClaw's managed local inference (`managed-inference/`, vLLM/llama.cpp
presets) is VRAM-bound — it serves what fits in physical VRAM. GreenBoost's
tiering serves a model *larger* than physical VRAM on the same box. That's
the actual reason a NemoClaw user would point at gb-synapse instead of
NemoClaw's own managed inference: more model, same hardware.

## Onboard recipe

1. Serve a model through gb-synapse as usual: `greenboost synapse run <model>`.
2. If NemoClaw's sandbox needs to reach it, the sandbox talks to the host over
   a docker bridge, **not loopback** — a plain `127.0.0.1`-bound gb-synapse
   (the default) will not be reachable from inside it. Serve with:
   ```bash
   export GB_SYNAPSE_BIND=0.0.0.0
   export GB_SYNAPSE_TOKEN=<secret>
   greenboost synapse run <model>
   ```
   A non-loopback bind with no token refuses to start (`gb_synapse_api.py`'s
   `main()`) — this is enforced, not just documented.
3. In NemoClaw: `nemoclaw onboard` → "Other OpenAI-compatible endpoint" →
   `http://<host>:11369/v1` (gb-synapse's default port,
   `GB_SYNAPSE_PORT` — see the collision note below for why it isn't
   `11435`) → model id from `GET /v1/models` (an id, not a file path) →
   `COMPATIBLE_API_KEY` = the `GB_SYNAPSE_TOKEN` value from step 2.
4. `synapse_doctor` (MCP) / `greenboost synapse doctor` (CLI) surfaces both
   halves of this before you start: whether `nemoclaw` is on `PATH`, and
   whether `GB_SYNAPSE_PORT` is already held by something that isn't
   gb-synapse itself.

## Why gb-synapse's default port is `11369`, not `11435`

gb-synapse used to default to `11435`. NemoClaw ships its own Ollama auth
proxy, and it defaults to that exact same port (`OLLAMA_PROXY_PORT` default
`11435`, `scripts/ollama-auth-proxy.mts`). Worse, `11435` sits in NemoClaw's
bundled host-gateway port allowlist alongside `8000`/`11434`, so on a box
running both, a custom endpoint pointed at `11435` gets silently rewritten
to `host.openshell.internal:11435` with no API-key prompt, and its
`local-inference` policy preset already allowlists that port into the
sandbox (`nemoclaw-blueprint/policies/presets/local-inference.yaml`). Two
things could end up serving the same port number with neither party
expecting the other.

**Fix (2026-08-05): gb-synapse's own default moved to `11369`** — closing
the collision on GreenBoost's side rather than asking every NemoClaw
install to move `OLLAMA_PROXY_PORT` instead. `GB_SYNAPSE_PORT=11435` still
works if you want the old value back (e.g. a box that's never running
NemoClaw), it's just no longer the default.

## Licensing note

GreenBoost is MIT; NemoClaw is Apache-2.0. Nothing in this document or in the
auth middleware it describes is adapted NemoClaw code — it's an independent
reimplementation of a design pattern (bearer-token proxy auth), which carries
no attribution obligation. If a future change forks actual NemoClaw
source (e.g. its `ServingRecipe` JSON Schema, audited but not yet adopted),
that file keeps its Apache-2.0 SPDX header and gets a
`third_party/nemoclaw_patterns/NOTICE` entry naming the source path and
commit — the same convention already used for `third_party/{llama.cpp,
turboquant,gemlite,auto_round,gb_cutlass}/NOTICE`.

## Round 2 (2026-08-06), the subsystems round 1 never opened

Round 1 mined one seam (inference/serving, uninstall, agent policy). Round 2
audited `src/lib/{advisories,shields,core,policy,readiness,diagnostics,state}`
and `scripts/checks/`, the same verdict holds (**NemoClaw stays a design
reference, never a runtime or a dependency**), design re-implemented from
scratch in Python/C/bash in every case below; nothing here needed a NOTICE
entry.

All 13 items landed:

| # | What | Where |
|---|---|---|
| 1 | Trust-validate `kernels.allow`/`cluster.conf`/`synapse_token` before honoring them (TCB-U1) | `greenboost_netd.c`'s `gb_trusted_root_file_fd()` |
| 2 | Atomic, locked JSON state writes | `gb_state_io.py` (`atomic_write_json`/`update_json_locked`) |
| 3 | Central port registry, collision-checked | `gb_ports.py`, `check_ports()` in the installer |
| 4 | One advisory contract for six producers | `gb_advisories.py` |
| 5 | Readiness qualifications + reference integrity | `gb_readiness.py` |
| 6 | Mechanical invariant checks, one command | `checks/run_checks.py` (`make check`), import boundaries, doc drift, installer parity, MCP parity, hardware-literal scan, secrets scan, dataflux coverage, vendor notices, semantics coverage, docs freshness |
| 7 | Closed-set failure classification with confidence downgrade | `gb_failures.py` |
| 8 | Bounded subprocess line decoding | `greenboost-cli/greenboost_cli/core/bounded_lines.py` (`mcp/client.py`, `cluster/peer_worker.py`) + a bounded-tail fix to `gb_synapse.py`'s `_upstream_log_tail()` (was `read_text()`-ing the whole log file) |
| 9 | Testable deadline+backoff waiting | `gb_wait.py` (`wait_until`), replaces the fixed while-loops in `gb_synapse.py`'s `_wait_upstream_ready`/`_wait_proxy_ready`/`wait_ready` |
| 10 | Innermost sub-stage progress labels | `gb_phase_activity.py` (Python) + `_gb_mark_phase_activity_push/pop` (bash, read by `gb_spin`), wired into HF pull, `quantize_to_fit`, feeder rsync, and the synapse torch engine's pip install |
| 11 | Redacted diagnostic support bundle | `gb_debug_bundle.py`, `greenboost debug bundle`, `support_bundle` MCP tool (dry-run by default) |
| 12 | Progressive MCP tool disclosure | `greenboost-cli/greenboost_cli/core/orchestrator.py`'s `_tool_search` / `ToolSearch` builtin |
| 13 | Installer/uninstaller parity as a mechanical check | `checks/check_installer_parity.py` + `checks/allowlists/install_manifest.txt`, a REVIEWED regex manifest at artifact granularity rather than a shared Python registry both `cmd_install_*`/`do_purge` import (13k-line bash made a fully declarative registry a much larger, riskier refactor for the same payoff: a check instead of a hand audit) |

**Found and fixed along the way, not in the original 13:** `gb_ports.py`,
`gb_advisories.py`, and `gb_failures.py` (items 3/4/7, landed earlier the
same round) were never added to `cmd_install_python_files`'s file list —
confirmed live, `import gb_readiness` against `$GB_PY_DEST` on a freshly
Full-Installed box raised `ModuleNotFoundError: No module named 'gb_ports'`.
Fixed in the same change. Separately, `.gitignore` had `checks/` and `docs/`
grouped under a "Claude agent skills/working files, never committed" block —
`docs/` has real git history (pre-existing commits) and is referenced
throughout CLAUDE.md as canonical, checked-in documentation; `checks/` is
this round's own item-6/13 deliverable. Both were accidental inclusions,
removed from `.gitignore`.

**Deliberately scoped down:** item 9's "calibrate the probe timeout from one
cheap request, record the discriminant" sub-feature (the backlog item the
port unblocks, not the port itself) was not built, `serving/probe.py`'s
`ProbeResult` is a tightly-scoped frozen dataclass with a closed failure-
reason set by design; bolting a new field on without a real consumer would
have been speculative. The core `wait_until` primitive is fully landed and
applied at all three real call sites.

Verification: `pytest tests/ -q` and `greenboost-cli`'s own suite both green
against a live Full Install (kmod loaded, shim attached), see
`workflow/known-issues.md` for the exact before/after counts. `checks/
run_checks.py --only import_boundaries,doc_drift,installer_parity,mcp_parity`
passes clean.
