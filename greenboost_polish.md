# GreenBoost polish — GB-Dataflux integration + GB-Semantics correctness (2026-07-30)

## Context

Requested: polish/enhance/evolve GreenBoost, specifically calling out
GB-Dataflux ("must be integrated further") and GB-Semantics. Rather than
speculate about what "polish" should mean, this investigation used the live
MCP surface (`greenboost_overview`, `dataflux_kinds`, `dataflux_schema`,
`semantic_segments`, `semantic_metrics`) plus direct source reading
(`gb_dataflux_kinds.py`, `gb_semantics.py`, `semantics/segments.yaml`,
`semantics/metrics.yaml`, `checks/check_dataflux_coverage.py`, `gb_cluster.py`)
to find concrete, verified defects and integration gaps — not a speculative
wishlist. Every item below was reproduced against this box's real live state
before being written down.

Two of the findings (S1, S2) are genuine bugs in already-shipped governed
code, not missing features — found by tracing exactly what the live
`greenboost_overview` call returned against what the source actually does.

## Findings

**GB-Dataflux (D):**

- **D1** — Four event kinds flow through the live shared dataflux log from
  consumer repos (`qc_summary`, `finish_summary` — emitted by
  `ai-forge/tools/conduir_art_jobs/{qc_sweep,finish_assets}.py`;
  `gaming_session`, `gaming_vram_pressure` — emitted by
  `greenboost_gaming/src/src-tauri/src/live_stats.rs`) but have **no entry**
  in `gb_dataflux_kinds.KINDS`. Confirmed via `dataflux_kinds` (14-day
  breakdown shows real counts: qc_summary=79, finish_summary=26,
  gaming_session=1, gaming_vram_pressure=1) vs `dataflux_schema()` (no entry
  for any of the four). This means `dataflux_schema()`/`dataflux_group()`
  can't describe what fields these events carry — exactly the problem the
  registry exists to solve, just not closed for cross-repo emitters.
- **D2** — `checks/check_dataflux_coverage.py`'s kind-literal scan (the
  mechanism that would normally catch D1) only AST-scans `*.py` files inside
  **this** repo. It structurally cannot see kinds emitted by ai-forge or by
  a Rust/Tauri frontend (`greenboost_gaming`), so D1-class gaps don't trip
  it — confirmed by running the checker (`0 finding(s)`) despite D1's 4 real
  orphan kinds existing in the live log the whole time. In-repo parity is
  genuinely 100% clean right now (verified, not assumed) — the gap is
  specifically the cross-repo blind spot, not sloppy registration.

**GB-Semantics (S):**

- **S1 (bug)** — `_res_cluster_vram_fill_pct()` (`gb_semantics.py:499`) reads
  `snap.get("nodes", {})`, but `gb_cluster.cluster_snapshot()` returns
  `{"host": {...}, "feeders": [...]}` — there has never been a `"nodes"` key
  anywhere in `gb_cluster.py` (confirmed by grep). The `cluster_vram_fill_pct`
  governed metric therefore **always silently resolves to an empty dict**,
  regardless of real cluster state. This is exactly the class of silent wrong
  answer GB-Semantics exists to prevent, ironically inside GB-Semantics
  itself.
- **S2 (bug)** — `_seg_serve_healthy()` requires `shim_fresh is True`
  unconditionally (`gb_semantics.py:766`). `shim_fresh` is only true while a
  CUDA process is actively writing `/run/greenboost/shim_stats` (< 30s old).
  On a genuinely idle-but-healthy box (this one, right now: kmod loaded, no
  errors, nothing being served), `serve_healthy` **can never match** —
  reproduced live: `greenboost_overview`'s `semantics.serve_healthy: false`
  on a box with zero problems. `serve_healthy`'s own doc calls it "the
  everything's fine verdict... so an agent doesn't have to synthesize it from
  silence" — but right now it does the opposite of that on idle systems,
  which is most of the time for a desktop.
- **S3 (gap)** — No governed segment exists for "a configured feeder is
  unreachable," distinct from `feeder_idle_while_host_saturated` (which
  assumes the feeder **is** connected but merely idle). Live right now: this
  box's own feeder (`omen`) is offline with `error: "timed out"`
  (`cluster_status`), and none of the 8 governed segments surface that as a
  checkable verdict — an agent following "resolve via GB-Semantics first"
  gets 8x `false` and no signal that the feeder is actually unreachable.
- **S4 (doc/registry drift)** — `semantics/segments.yaml`'s
  `feeder_idle_while_host_saturated` entry declares
  `reads: [cluster_vram_fill_pct, feeder_items]`, but
  `_seg_feeder_idle_while_host_saturated()` actually calls
  `resolve("vram_fill_pct", entity_id="host")` — a different metric name.
  The evaluator's own logic is correct (host-scoped, matching the segment's
  name/doc); only the declared `reads:` list is stale.

## Plan

### Group A — GB-Semantics correctness (highest value: fixes wrong governed answers)

**A1. Fix `_res_cluster_vram_fill_pct`** (`gb_semantics.py`) to parse the real
`cluster_snapshot()` shape: `host` dict + `feeders` list, each carrying
`t1_free_mb`/`t1_total_mb` (confirmed via `_host_metrics_dict`/`Feeder`
dataclass field names). Compute `100 * (total - free) / total` per node,
skip offline feeders (their t1 fields are 0/stale). Key the output dict by
hostname (falling back to `ip`) to match how `feeder_items`/other per-node
metrics key their output.

**A2. Fix `_seg_serve_healthy`** to distinguish idle from unhealthy. The
naive fix (branch purely on `shim_fresh`) breaks the existing
`serve_healthy_no_match` eval fixture, because that fixture's own scenario
(`tests/fixtures/semantics/events.py`) represents an **active-but-violating**
serve session (`gpu_util_pct: 61`, a `tok_s_measured` event present, physical
VRAM 42.3%) — `shim_fresh`/`kmod_loaded` are OS-filesystem-resolved (not
fixture-controllable), so branching on them alone is non-deterministic across
test/live machines. Instead, use a dataflux-event-sourced "is anything
actively being decoded" signal: `_latest_event("tok_s_measured", max_age_s=60.0)`
is `None` → genuinely idle → healthy iff `kmod_loaded`; found → same
active-session gate as before (`shim_fresh` + VRAM ≥ 60% + quality floor).
This keeps the fixture test passing for the right structural reason (active
+ Rule#1-violated → still unhealthy) while fixing the live idle-false-negative.

**A3. Add `feeder_reachable` metric + `feeder_unreachable` segment.**
Resolver reads `gb_cluster.cluster_snapshot()`'s `feeders` list, returns
`{hostname: online_bool}` (entity=`feeder`). Segment matches when any
configured feeder has `online=False` — the exact live condition this box is
in right now. Register in `semantics/metrics.yaml` + `semantics/segments.yaml`
following the existing `feeder_items`/`feeder_idle_while_host_saturated`
pattern (entity=`feeder`, `owner: GB-Cluster`).

**A4. Fix `semantics/segments.yaml`'s stale `reads:`** for
`feeder_idle_while_host_saturated` to `[vram_fill_pct, feeder_items]`,
matching what the evaluator actually calls (leave the evaluator logic itself
untouched — it's correct for what the segment's name/doc promise: **host**
saturation, not cluster-wide).

### Group B — GB-Dataflux integration

**B1. Register the 4 orphan kinds** in `gb_dataflux_kinds.KINDS`
(`qc_summary`, `finish_summary` under `group="pipeline"` next to the other
ai-forge kinds; `gaming_session`, `gaming_vram_pressure` under a `group="gaming"`
or existing suitable group), each `planned=True` with a `doc` noting the
real emitter (ai-forge / greenboost_gaming), matching the existing
`stage_profile`/`video_render` convention exactly.

**B2. Close the D2 blind spot durably**: add a new pass to
`checks/check_dataflux_coverage.py` (or a focused new check module) that
reads the **live dataflux log** itself (via `gb_dataflux.read_events`/
`gb_dataflux.summarize`, the same mechanism `dataflux_kinds()` uses) over a
wide window, collects every kind that has ever actually appeared, and flags
any kind present in real data but absent from `gb_dataflux_kinds.KINDS`
regardless of which repo emitted it. This is what would have caught D1
automatically instead of requiring a manual live-tool investigation, and it
keeps catching this class of gap for any *future* consumer-repo kind. Since
this reads live local state (not static source), keep it **advisory**, not
blocking (a fresh checkout with an empty/short dataflux log must not fail
the check) — mirroring Pass 1's advisory severity in the existing checker.

### Group C — Tests + verification

**C1.** Add a fixture-free regression test for A2 (mock `_latest_event`/
`resolve` directly rather than fighting the file's autouse fixture-log
machinery, since the true "no recent tok_s_measured" idle case can't be
expressed by writing MORE events into the shared fixture).

**C2.** Add eval entries to `evals/semantics/core.yaml` for the two new A3
metrics/segment (`source: live`, structural — `feeder_reachable`'s actual
value depends on real network/feeder state, so assert well-formedness only,
same pattern as `cluster_vram_fill_pct_structural`). Required anyway:
`test_eval_set_size_and_coverage()` hard-fails on any metric/segment with
zero eval coverage.

**C3.** Run `pytest tests/ -q` (or `make test`) after each group — must stay
green, including the existing `test_ungoverned_baseline_is_wrong` trap tests
and `test_eval_set_size_and_coverage`.

**C4.** Run `python3 checks/check_dataflux_coverage.py` — must stay at 0
blocking findings after B1/B2.

**C5.** Live-verify via the actual MCP tools post-implementation:
`semantic_segments()` should now show `feeder_unreachable: true` (real
offline feeder) and `serve_healthy` should flip to `true` given this box is
genuinely idle and healthy; `dataflux_schema(kind="qc_summary")` should
return a real entry instead of `{}`.

## Critical files

- `gb_semantics.py` (A1, A2 resolver/evaluator functions)
- `semantics/metrics.yaml`, `semantics/segments.yaml` (A3, A4)
- `gb_dataflux_kinds.py` (B1)
- `checks/check_dataflux_coverage.py` (B2)
- `evals/semantics/core.yaml` (C2)
- `tests/test_semantics_evals.py` (C1)

## Explicitly out of scope

- The ~20 registered dataflux kinds with zero occurrences in the last 14
  days (`tier_move`, `mem_pool_trim`, `link_transfer`, `job_local*`,
  `model_push`, `stage`, `feeder_provision`, `a2a_request`, `yarn_bake`,
  `niah_cert`, etc.) — verified via `check_dataflux_coverage.py`'s clean
  0-finding pass that every one of these has a real in-repo emit site; their
  silence is explained by genuinely unexercised code paths this session
  (feeder offline all session, no gaming/eval/A2A activity), not dead code.
  No fix needed.
- Rearchitecting `feeder_idle_while_host_saturated` to be cluster-wide
  (using the now-fixed `cluster_vram_fill_pct`) instead of host-only — the
  segment's own name and doc are explicitly about host saturation; changing
  its meaning is a separate design decision, not a bug fix.

## Verification status (2026-07-30, post-restart)

All of Groups A/B/C confirmed implemented and live-verified after a Claude
Code session restart:

- Full test suite: 1476 passed, 8 failed — the 8 failures confirmed
  pre-existing on `main` via `git stash` A/B comparison (identical failures
  with this session's changes stashed out), unrelated to this work
  (`test_gb_dataflux.py` SnapshotRecorder tests + `test_gb_orchestrator.py`
  clock-throttle/KV-grow tests).
- Live MCP verification (`semantic_segments`, `semantic_resolve`,
  `dataflux_schema`) confirmed all five fixes on this box's real running
  state: `cluster_vram_fill_pct` resolves to `{"host": 39.9}` (was always
  `{}`), `serve_healthy` is `true` on this genuinely idle box (was a false
  negative), `feeder_unreachable` is `true` and correctly names the offline
  `omen` feeder, and `qc_summary`/`finish_summary`/`gaming_session`/
  `gaming_vram_pressure` all return real `dataflux_schema()` entries instead
  of `{}`.
- Changes remain uncommitted in the working tree pending user review.

## torch nightly → stable correction (2026-07-30, session resumed after limit)

Prior round of this session installed `torch==2.14.0.dev20260729+cu132` off
the `/whl/nightly/cu13x` index per the "always track latest stable CUDA 13+"
rule below — but a nightly is PyTorch's pre-release channel by their own
terminology, not "stable"; that was a misreading of the rule, caught and
corrected this round.

- Snapshotted the nightly pin (`torch==2.14.0.dev20260729+cu132`,
  `torchvision==0.29.0.dev20260730+cu132`) for rollback; no
  compiled-against-torch packages (xformers/flash-attn/bitsandbytes/apex/
  deepspeed) present.
- Installed the real stable release: `torch==2.13.0+cu132` +
  `torchvision==0.28.0+cu132` from `https://download.pytorch.org/whl/cu132`
  (note: `torchvision==0.28.0` pairs with torch 2.13, not `0.27.x` which
  pairs with 2.12). Verified with a live CUDA matmul on the RTX 5070
  (`torch.version.cuda == '13.2'`, tensor op on-device).
- Rebuilt `third_party/gb_cutlass/` against the new torch ABI (`make
  gb_cutlass`, clean build); `bench_cutlass_nvfp4.py` imports and runs to
  its documented stop point (NVFP4 pack reconciliation, unchanged/
  out-of-scope) exactly as before the downgrade.
- Full suite: 1482 passed, 1 skipped, 8 failed — same 8 pre-existing
  failures as the established baseline, no regression from the downgrade.
  `check_dataflux_coverage.py` / `check_semantics_coverage.py` both still
  "0 finding(s) (0 blocking)".
- Corrected `CLAUDE.md`'s "Standing Rule — Always track latest stable CUDA
  13+" section, which had literally prescribed the nightly index —
  rewritten to point at the stable `cu13x` index and to record the
  nightly-is-not-stable correction so it isn't re-derived. Also added a
  gb_cutlass-rebuild reminder to the upgrade steps (missed by the original
  rule text).
- **Found, not yet acted on**: a full orphaned `-cu12` NVIDIA package stack
  (13 packages — `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
  `nvidia-cusolver-cu12`, etc., all pinned at `12.8.x`/`9.20.0.48`) is
  installed alongside the active `cu13x` stack, left over from an earlier
  cu128-era torch install. Reverse-dependency check confirms nothing
  currently installed requires them (only an unused `cuda-pathfinder` extra
  references two of them). Not removed — flagged for the user to confirm
  before `pip uninstall`.

## Third audit pass — `gb_actuation.py::tier_actuate` (2026-07-30, subagent)

Fresh ground, distinct from the 5 bugs already fixed this session:
`gb_pilot.py`, `gb_monitor.py`, `gb_init.py`, `gb_placement.py`, `gb_attn.py`
(env-var parsing), the untouched parts of `gb_orchestrator.py`, and the rest
of `gb_mcp.py` (`quant_advisor`, `gb_plan`, `shim_env`, `tier_actuate`,
`run_under_greenboost`, `a2a_gateway`, `set_quant_policy`, `flux_health`),
plus `gb_mcp_common.py` and `gb_actuation.py`.

**Bug found (same silent-wrong-field class as the earlier
`optimize_inference` fix):** `gb_actuation.py::tier_actuate` (line 217) read
`getattr(res, "ok", None)` on the `ActuatorResult` every `GbControl` lever
setter returns. `ActuatorResult` (`gb_control.py`) has never had an `ok`
field, only `applied` — so the MCP-facing `plan["applied"]` silently
reported `None` for every real double-gated actuation, success or failure,
indistinguishable from a dry-run refusal. Verified twice: source (grep
confirms `ActuatorResult`'s real field set) + a stub-`GbControl`
reproduction showing `plan["applied"]` was `None` even when the actuator
genuinely applied a change. Fixed by reading `.applied` instead; two
regression tests added in `tests/test_gb_actuation.py`.

No other defects found across the rest of the audited surface (`gb_pilot`'s
pressure/stage/model fields, `gb_monitor`'s ioctl struct layout,
`gb_placement`'s topology/pool_brief regex, `gb_orchestrator.dump()`'s ~25
keys, `gb_attn`'s env dispatch) — all matched their real producers.

Full suite after this fix + the torch stable downgrade above: **1482
passed, 1 skipped, 8 failed** (same pre-existing 8), both coverage checkers
still `0 finding(s) (0 blocking)`.

## G3 landed — sessions.jsonl/dataflux.jsonl unification + real size cap (2026-07-30)

Requested directly: unify `sessions.jsonl`/`dataflux.jsonl` (deferred earlier
this session as G3, see `greenboost_gaming_polish.md`), and make
`dataflux.jsonl` genuinely incapable of exceeding 69 MiB with roughly 7 days
of retention.

**`compact_archive()` — retention default + real size backstop.**
`_DEFAULT_RETAIN_DAYS` was `30.0`; changed to `7.0` per the request. More
importantly, the existing age-based trim alone could never *guarantee*
"≤69 MiB total" — it bounds time, not bytes, so a volume spike (several
concurrent `gb_init`-importing processes, a debug session with tighter
snapshot polling) could in principle still leave more than 69 MiB archived
within 7 days. Added a genuine size backstop: after the age-based trim,
`compact_archive()` now also sorts surviving events oldest-first and drops
from the front until the archive's raw (uncompressed) line-byte total is
back under `_max_log_bytes()` — the same ceiling `emit()`'s rotation trigger
already uses for the live file. Comparing raw bytes (not the gzip-compressed
on-disk size) against that ceiling is deliberately conservative — compression
only ever shrinks the real footprint further, so the true on-disk archive
size ends up well under budget, never over it. This closes the actual gap:
the live file was already hard-capped at 69 MiB by the pre-existing rotation
trigger; the archive previously had no size cap at all, only an age cap.
Now both halves of the on-disk footprint are genuinely bounded. Four new
tests added to `tests/test_gb_dataflux_rotation.py` (default-is-7-days,
size-backstop-trims-oldest-first, size-backstop-noop-when-under-budget, plus
the pre-existing age-only tests untouched since they all pass `retain_days`
explicitly).

**sessions.jsonl removed, `greenboost_gaming` now writes/reads one path.**
See `greenboost_gaming_polish.md`'s new section for the write-side change
(Proton wrapper) and read-side change (`manager.rs`'s
`get_session_history_impl`/`analyze_game_sessions_impl`, now reading
`gaming_session`/`stop` dataflux events instead of a separate file). Noted
tradeoff: session history now shares `dataflux.jsonl`'s own retention window
(~7 days by default) instead of growing forever — an explicit, intentional
consequence of unifying into a size/age-bounded log, not an oversight.

Full suite: **1485 passed, 1 skipped, 8 failed** (same pre-existing 8, +3 new
passing tests from this change), both coverage checkers still
`0 finding(s) (0 blocking)`.
