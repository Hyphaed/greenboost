# A2A interop — two gateways, one cluster

GreenBoost's cluster has **two** A2A (agent-to-agent) gateways, serving
different concerns. This is deliberate, not drift — each gateway is the
right shape for its job, and they don't compete.

| | `gb_a2a.py` (this repo) | `studio/server/a2a_gateway.py` (ai-forge) |
|---|---|---|
| Concern | **Node-level actuation** — GreenBoost tiering/quant/cluster/serving levers | **Task delegation** — route a generation job to the best cluster node |
| Protocol | Hand-rolled JSON-RPC 2.0, stdlib `http.server` | a2a-sdk v0.3 (`a2a.compat.v0_3`), FastAPI |
| Discovery | `GET /.well-known/agent.json` (legacy shape) **and** `GET /.well-known/agent-card.json` (v0.3 shape, see below) | `GET /.well-known/agent-card.json` (gateway card) + `GET /a2a/nodes/{node}` (per-node card) |
| Verbs / skills | `gb_actuation.VERBS`: `cluster_ensure_feeder_ready`, `cluster_dispatch_plan`, `set_quant_policy`, `tier_actuate`, `serve_and_repoint`, `shim_env`, `run_under_greenboost` | Generation-job skills per node (`node_inventory.SKILL_META`) |
| Gate | Double-gated: `confirm=True` AND `GB_ORCH_ACTUATE=1`. A2A cannot bypass either half. | ai-forge's own job-queue admission logic |
| Bind | `127.0.0.1:8790` default; LAN requires `GB_A2A_TOKEN` | ai-forge's `PORT` (studio config) |
| Runs as | `greenboost-a2a.service` (systemd, installed + `enable --now` by Full Install, loopback + actuation-off by default) | part of the ai-forge studio server process |

## Why not merge them

`gb_a2a.py` exists to let a delegating agent **change GreenBoost's own
configuration** (grow a T2 pool, re-quantize, provision a feeder, run a
command under the shim) — the same gated verb table the MCP tools use, so
neither control plane can do something the other can't. It has no concept of
"a generation job."

ai-forge's gateway exists to let a delegating agent **submit work** (image/
video/animation generation) and have it land on whichever cluster node is
best-suited (idle, has the model, fastest link) — it has no concept of
"actuate a GbControl lever." It calls into `gb_cluster.run_stage_on_feeder`
as a **consumer**, the same way any other ai-forge pipeline does.

Collapsing these into one gateway would mean either the node-actuation
surface grows a job-queue (out of scope for a memory-tiering project) or the
job-delegation surface grows GbControl levers (out of scope for a pipeline
orchestration project). Keeping them separate keeps each one legible.

## Discovery convention

Both gateways serve `GET /.well-known/agent-card.json` (the A2A protocol
v0.3 well-known path) so a generic A2A-aware client can discover either one
without knowing in advance which kind of agent it's talking to — the card's
`name`/`description`/`skills` disambiguate. `gb_a2a.py` ALSO serves the
legacy `GET /.well-known/agent.json` path (its original shape, predating the
v0.3 card) for backward compatibility with anything already pointed at it.

`gb_actuation.agent_card()` builds the legacy shape; `gb_actuation.agent_card_v03()`
derives the v0.3-shaped card from the same underlying data (same skills, same
nodes) — one source of truth, two JSON shapes. If you add a verb to
`gb_actuation.VERBS`, both cards pick it up automatically; you never need to
touch the card-building code itself.

## Interop caveat

`agent_card_v03()` is hand-built JSON matching the A2A v0.3 spec's documented
field names (`protocolVersion`, `preferredTransport`, `defaultInputModes`,
etc.) — `gb_a2a.py` stays a stdlib-only server (no `a2a-sdk` dependency), so
this has NOT been validated against the installed `a2a-sdk` package's strict
schema. If a client validates strictly against the SDK's pydantic models and
rejects the card, compare field-by-field against
`ai-forge/studio/server/a2a_gateway.py`'s `_gateway_card()`/`_node_card()`
(which DOES import the SDK) and adjust.

## Operating both

```bash
# GreenBoost's own gateway (node actuation)
systemctl status greenboost-a2a.service
curl -s http://127.0.0.1:8790/.well-known/agent-card.json | python3 -m json.tool

# Enable real actuation (default: observe/dry-run only)
sudo systemctl edit greenboost-a2a.service   # add Environment=GB_ORCH_ACTUATE=1
sudo systemctl restart greenboost-a2a.service

# Query gateway liveness + recent requests from an LLM/agent
# (greenboost-dataflux MCP): a2a_status
# Restart/status via systemd from an LLM/agent
# (greenboost-orchestrator MCP): a2a_gateway(action="status"|"restart")

# ai-forge's gateway (task delegation) — see ai-forge/docs/A2A-DESIGN.md
```
